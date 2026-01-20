"""Goal inference engine - converts recorded actions into goal-oriented steps.

Key insight: Look at OUTCOMES, not just ACTIONS.
- What URL did we end up at after a click?
- What data did we copy?
- What page type did we reach?

This allows us to infer the user's INTENT and create robust goal steps.

FUNDAMENTAL PRINCIPLE (v2):
Use LLM at COMPILE TIME to understand INTENT, not just record actions.
- A click followed by typing is ONE "search" action, not two separate steps
- A navigation to youtube.com might be intentional OR incidental
- The LLM analyzes the CONTEXT to determine flexible vs rigid success criteria
"""
from typing import List, Optional, Dict, Any, Tuple
import uuid
import json
from urllib.parse import urlparse

from src.models.semantic_trace import SemanticStep, VoiceContext
from src.models.goal_step import (
    GoalStep, GoalType, SuccessCriteria, Strategy, GoalWorkflow
)
from src.models.workflow_recipe import (
    WorkflowRecipe, WorkflowStep, WorkflowParameter,
    ExtractionSchema, FailurePolicy
)
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client


class GoalInferrer:
    """
    Infers high-level goals from recorded actions.
    
    Process:
    1. Pre-analyze step sequences using LLM to understand intent
    2. Merge behavioral sequences (click+type → search)
    3. Determine flexible vs rigid success criteria
    4. Create multiple strategies to achieve the goal
    """
    
    def __init__(self):
        self.logger = setup_logger("GoalInferrer")
        # Parameter mapping for template creation
        self._param_value_to_name: Dict[str, str] = {}
        self._param_templates: Dict[str, str] = {}
        # Cache for LLM step analysis results
        self._step_analysis_cache: Dict[str, Dict[str, Any]] = {}
        # Track which steps should be skipped (merged into others)
        self._steps_to_skip: set = set()
    
    # =========================================================================
    # LLM-BASED STEP ANALYSIS (FUNDAMENTAL FIX)
    # =========================================================================
    
    def _analyze_step_sequence_with_llm(
        self,
        steps: List[SemanticStep],
        voice_context: Optional[VoiceContext] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Use LLM to analyze the entire step sequence and classify intent.
        
        Returns a dict mapping step_id -> analysis results including:
        - is_focus_click: True if click is just to focus an input (merge with next type)
        - navigation_intent: 'specific_site' | 'any_result' | 'first_result'
        - should_skip: True if this step should be merged into another
        - merge_with: step_id to merge this step into (if applicable)
        """
        if not llm_client.is_available:
            self.logger.warning("LLM not available, using heuristic analysis")
            return self._heuristic_step_analysis(steps, voice_context)
        
        # Build step summary for LLM
        steps_summary = []
        for i, step in enumerate(steps):
            summary = {
                "index": i,
                "step_id": step.step_id,
                "intent": step.intent,
                "app": step.app_name,
                "url": step.url_before,
                "typed": step.typed_values,
                "clicked": [e.get("text", "")[:50] for e in step.clicked_elements] if step.clicked_elements else [],
                "shortcuts": step.keyboard_shortcuts,
                "time_since_prev": step.start_timestamp - steps[i-1].end_timestamp if i > 0 else 0
            }
            steps_summary.append(summary)
        
        voice_hint = ""
        if voice_context:
            # Build voice hint from available context
            hints = []
            if voice_context.task_goal:
                hints.append(f"Task goal: {voice_context.task_goal}")
            if voice_context.extraction_hints:
                hints.append(f"Extraction hints: {', '.join(voice_context.extraction_hints)}")
            if voice_context.parameter_hints:
                for ph in voice_context.parameter_hints[:5]:
                    hints.append(f"Mentioned: {ph.get('value', '')} (type: {ph.get('type', '')})")
            if hints:
                voice_hint = f"\nUser voice context: {'; '.join(hints)}"
        
        prompt = f"""Analyze this recorded user workflow to understand INTENT, not just actions.

Steps recorded:
{json.dumps(steps_summary, indent=2)}
{voice_hint}

For each step, determine:
1. Is this a "focus click" (clicking to focus an input before typing)? 
   - Key indicator: click followed by typing within 2 seconds on same app
   
2. For navigation clicks (click that changes URL/domain):
   - Is this "intentional navigation" (user specifically wanted THIS site)?
   - Or "incidental navigation" (user clicked the first/best result, any good result works)?
   - Check voice narration - if user mentioned a specific site, it's intentional
   - If query contains site name (e.g., "zomato", "youtube"), it's intentional
   
3. Should this step be merged with another? (focus_click → merge into next search/type)

Return JSON:
{{
    "steps": {{
        "<step_id>": {{
            "is_focus_click": boolean,
            "navigation_intent": "specific_site" | "any_result" | "first_result" | null,
            "target_site": "domain if specific_site, else null",
            "should_skip": boolean,
            "merge_into": "step_id to merge into, or null",
            "reasoning": "brief explanation"
        }}
    }},
    "workflow_summary": "brief description of what user was trying to accomplish"
}}

Be generous with "any_result" - if there's no strong evidence of specific site intent, use "any_result"."""

        try:
            result = llm_client.complete_json(prompt, temperature=0.1)
            if result and "steps" in result:
                self.logger.info(f"LLM analyzed {len(result['steps'])} steps")
                self.logger.debug(f"Workflow summary: {result.get('workflow_summary', 'N/A')}")
                return result["steps"]
        except Exception as e:
            self.logger.error(f"LLM step analysis failed: {e}")
        
        return self._heuristic_step_analysis(steps, voice_context)
    
    def _heuristic_step_analysis(
        self,
        steps: List[SemanticStep],
        voice_context: Optional[VoiceContext] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fallback heuristic analysis when LLM is unavailable.
        Uses time-based and pattern-based detection.
        """
        analysis = {}
        
        for i, step in enumerate(steps):
            next_step = steps[i + 1] if i + 1 < len(steps) else None
            
            # Detect focus click: click followed by typing within 2s
            is_focus_click = False
            merge_into = None
            
            if step.intent == "select" and next_step:
                time_gap = next_step.start_timestamp - step.end_timestamp
                same_app = step.app_name == next_step.app_name
                next_is_type = next_step.intent in ["search", "write"] and next_step.typed_values
                
                if same_app and next_is_type and time_gap < 2.0:
                    is_focus_click = True
                    merge_into = next_step.step_id
            
            # Detect navigation intent
            navigation_intent = None
            target_site = None
            
            if step.intent == "select":
                # Check if query contains site name
                query = ""
                for prev_step in steps[:i]:
                    if prev_step.intent == "search" and prev_step.typed_values:
                        query = " ".join(prev_step.typed_values).lower()
                        break
                
                site_indicators = ["zomato", "yelp", "youtube", "amazon", "flipkart", 
                                  "swiggy", "booking.com", "tripadvisor", "google maps"]
                
                for site in site_indicators:
                    if site in query:
                        navigation_intent = "specific_site"
                        target_site = site
                        break
                
                if not navigation_intent:
                    navigation_intent = "any_result"
            
            analysis[step.step_id] = {
                "is_focus_click": is_focus_click,
                "navigation_intent": navigation_intent,
                "target_site": target_site,
                "should_skip": is_focus_click,
                "merge_into": merge_into,
                "reasoning": "heuristic analysis"
            }
        
        return analysis
    
    def infer_goals_from_semantic_steps(
        self,
        semantic_steps: List[SemanticStep],
        voice_context: Optional[VoiceContext] = None,
        extraction_schemas: Optional[Dict[str, Any]] = None,
        detected_parameters: Optional[List[Any]] = None
    ) -> List[GoalStep]:
        """
        Convert semantic steps into goal-oriented steps.
        
        Args:
            semantic_steps: Steps from timeline segmentation
            voice_context: Voice narration context (if available)
            extraction_schemas: Gemini-enriched extraction schemas
            detected_parameters: Parameter candidates from ParameterDetector
        
        Returns:
            List of GoalSteps with inferred goals and strategies
        """
        goal_steps = []
        extraction_schemas = extraction_schemas or {}
        
        # Build parameter value -> name mapping for template creation
        self._param_value_to_name = {}
        self._param_templates = {}
        if detected_parameters:
            for param in detected_parameters:
                if param.confidence >= 0.5:
                    self._param_value_to_name[param.value.lower()] = param.suggested_name
                    self.logger.debug(f"  Param: {param.suggested_name} = '{param.value}'")
        
        # =====================================================================
        # FUNDAMENTAL FIX: Pre-analyze steps with LLM to understand intent
        # =====================================================================
        self.logger.info("Analyzing step sequence with LLM...")
        self._step_analysis_cache = self._analyze_step_sequence_with_llm(
            semantic_steps, voice_context
        )
        self._steps_to_skip = set()
        
        # Mark steps that should be skipped (merged into others)
        for step_id, analysis in self._step_analysis_cache.items():
            if analysis.get("should_skip"):
                self._steps_to_skip.add(step_id)
                self.logger.debug(f"  Will skip step {step_id} (merged into {analysis.get('merge_into')})")
        
        for i, step in enumerate(semantic_steps):
            # Skip steps that were merged into others (e.g., focus clicks)
            if step.step_id in self._steps_to_skip:
                self.logger.debug(f"Skipping step {step.step_number} (merged)")
                continue
            
            # Get context from surrounding steps
            prev_step = semantic_steps[i - 1] if i > 0 else None
            next_steps = semantic_steps[i + 1: i + 4]  # Look ahead 3 steps
            
            # Analyze what happened after this step
            outcome = self._analyze_outcome(step, next_steps)
            
            # Get LLM analysis for this step
            step_analysis = self._step_analysis_cache.get(step.step_id, {})
            
            # =====================================================================
            # BUNDLED SHORTCUTS FIX: Extract paste/save from app_switch events
            # =====================================================================
            # If this is an app_switch with paste/save, those happened in PREVIOUS app
            if (step.boundary_reason == "app_switch" and 
                step.keyboard_shortcuts and 
                any(s in step.keyboard_shortcuts for s in ["paste", "save"])):
                
                prev_app = prev_step.app_name if prev_step else None
                bundled_goals = self._extract_bundled_shortcuts(step, prev_app)
                goal_steps.extend(bundled_goals)
            
            # Infer the goal (now with LLM analysis context)
            goal = self._infer_goal_from_step(
                step=step,
                step_index=i,
                all_steps=semantic_steps,
                outcome=outcome,
                voice_context=voice_context,
                extraction_schemas=extraction_schemas,
                step_analysis=step_analysis
            )
            
            if goal:
                goal_steps.append(goal)
                self.logger.info(
                    f"Step {step.step_number}: {step.intent} → Goal: {goal.goal_type.value} "
                    f"({goal.goal_description[:50]}...)"
                )
        
        # =====================================================================
        # STEP CONSOLIDATION: Reduce noise and redundancy
        # =====================================================================
        original_count = len(goal_steps)
        goal_steps = self._consolidate_goals(goal_steps)
        
        if len(goal_steps) < original_count:
            self.logger.info(f"  Consolidated {original_count} → {len(goal_steps)} goals")
        
        return goal_steps
    
    def _consolidate_goals(self, goals: List[GoalStep]) -> List[GoalStep]:
        """
        Consolidate redundant/spurious goals using semantic analysis.
        
        This is GENERIC - no hardcoded rules about specific apps or websites.
        
        Consolidation rules (in order):
        1. Remove duplicate extracts (keep first meaningful one)
        2. Remove spurious clicks (no outcome, generic description)
        3. Merge consecutive same-app launches into one
        4. Remove redundant navigate goals (already on target)
        5. Merge redundant write goals (same template/input)
        """
        if not goals:
            return goals
        
        consolidated = []
        seen_extracts = 0  # Count meaningful extracts
        last_launch_app = None
        last_meaningful_select = None
        
        for i, goal in enumerate(goals):
            prev_goal = consolidated[-1] if consolidated else None
            
            # -----------------------------------------------------------------
            # Rule 1: Skip duplicate EXTRACT goals (keep first with schema)
            # -----------------------------------------------------------------
            if goal.goal_type == GoalType.EXTRACT:
                # Keep if this is the first extract with a schema
                has_schema = bool(goal.extraction_schema)
                if seen_extracts == 0 or has_schema:
                    consolidated.append(goal)
                    seen_extracts += 1
                    self.logger.debug(f"  Keeping extract #{seen_extracts}")
                else:
                    self.logger.debug(f"  Skipping duplicate extract (already have {seen_extracts})")
                continue
            
            # -----------------------------------------------------------------
            # Rule 2: Skip spurious SELECT/click goals (generic, no outcome)
            # -----------------------------------------------------------------
            if goal.goal_type == GoalType.SELECT:
                # Check if this is a "spurious" click
                if self._is_spurious_select(goal):
                    self.logger.debug(f"  Skipping spurious select: {goal.goal_description[:40]}")
                    continue
                
                # Check if it's similar to the last meaningful select
                if last_meaningful_select and self._are_similar_selects(last_meaningful_select, goal):
                    self.logger.debug(f"  Skipping similar select: {goal.goal_description[:40]}")
                    continue
                
                last_meaningful_select = goal
            
            # -----------------------------------------------------------------
            # Rule 3: Merge consecutive LAUNCH goals for same app
            # -----------------------------------------------------------------
            if goal.goal_type == GoalType.LAUNCH:
                target_app = goal.app_name
                if target_app == last_launch_app:
                    self.logger.debug(f"  Skipping duplicate launch: {target_app}")
                    continue
                
                # Also skip if previous goal is already in this app
                if prev_goal and prev_goal.app_name == target_app:
                    self.logger.debug(f"  Skipping unnecessary launch (already in {target_app})")
                    continue
                
                last_launch_app = target_app
            else:
                # Reset launch tracking when we do something else
                last_launch_app = goal.app_name
            
            # -----------------------------------------------------------------
            # Rule 4: Skip NAVIGATE if we're already navigating to same target
            # -----------------------------------------------------------------
            if goal.goal_type == GoalType.NAVIGATE:
                if prev_goal and prev_goal.goal_type == GoalType.NAVIGATE:
                    # Check if same target
                    prev_target = prev_goal.success_criteria.url_contains or ""
                    curr_target = goal.success_criteria.url_contains or ""
                    if prev_target and curr_target and prev_target.lower() == curr_target.lower():
                        self.logger.debug(f"  Skipping duplicate navigate: {curr_target}")
                        continue
            
            # -----------------------------------------------------------------
            # Rule 5: Merge WRITE goals that are just labels/annotations
            # -----------------------------------------------------------------
            if goal.goal_type == GoalType.WRITE:
                # Skip very short writes (single chars, spaces)
                input_val = goal.parameters.get("text", "") or goal.template or ""
                if len(input_val.strip()) <= 2:
                    self.logger.debug(f"  Skipping trivial write: '{input_val}'")
                    continue
            
            # -----------------------------------------------------------------
            # Rule 6: Skip redundant SHORTCUT goals (multiple pastes)
            # -----------------------------------------------------------------
            if goal.goal_type == GoalType.SHORTCUT:
                # Keep first paste, skip subsequent ones unless they're different
                is_paste = "paste" in goal.goal_description.lower()
                if is_paste and prev_goal and prev_goal.goal_type == GoalType.SHORTCUT:
                    prev_is_paste = "paste" in prev_goal.goal_description.lower()
                    if prev_is_paste:
                        # Check if templates are same
                        if prev_goal.template == goal.template:
                            self.logger.debug(f"  Skipping duplicate paste")
                            continue
            
            # Add goal to consolidated list
            consolidated.append(goal)
        
        return consolidated
    
    def _is_spurious_select(self, goal: GoalStep) -> bool:
        """
        Detect if a SELECT goal is spurious (noise, not meaningful).
        
        Generic detection based on:
        1. No meaningful outcome (no URL change expected)
        2. Very generic description
        3. No specific element identification
        """
        criteria = goal.success_criteria
        
        # If it expects URL change OR element visibility, it's not spurious
        if criteria.url_changed and not criteria.timeout_success:
            return False
        
        if criteria.element_visible or criteria.page_type:
            return False
        
        # Check if description is generic
        desc = goal.goal_description.lower()
        generic_patterns = [
            "click element in",
            "click in",
            "clickable element",
            "interactive element",
        ]
        is_generic = any(p in desc for p in generic_patterns)
        
        # Check if we have specific element info
        has_specific_strategy = False
        for strategy in goal.strategies:
            if strategy.selector or strategy.text_match:
                has_specific_strategy = True
                break
            if strategy.visual_description and "first" not in strategy.visual_description:
                has_specific_strategy = True
                break
        
        # It's spurious if generic AND no specific strategy AND timeout_success
        return is_generic and not has_specific_strategy and criteria.timeout_success
    
    def _are_similar_selects(self, goal1: GoalStep, goal2: GoalStep) -> bool:
        """
        Check if two SELECT goals are essentially the same.
        
        Similar if:
        - Same app
        - Same or similar description
        - Same coordinates (within tolerance)
        """
        if goal1.app_name != goal2.app_name:
            return False
        
        # Check if descriptions are very similar
        desc1 = goal1.goal_description.lower()
        desc2 = goal2.goal_description.lower()
        if desc1 == desc2:
            return True
        
        # Check coordinates
        coords1 = None
        coords2 = None
        for s in goal1.strategies:
            if s.coordinates:
                coords1 = s.coordinates
                break
        for s in goal2.strategies:
            if s.coordinates:
                coords2 = s.coordinates
                break
        
        if coords1 and coords2:
            # Within 50px tolerance
            if abs(coords1[0] - coords2[0]) < 50 and abs(coords1[1] - coords2[1]) < 50:
                return True
        
        return False
    
    def _create_template_from_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Create a template by replacing detected parameter values with placeholders.
        
        Args:
            text: Original typed text like "best bars in delhi zomato"
        
        Returns:
            Tuple of (template, parameters_dict)
            e.g., ("{{query}} in {{location}} {{site_filter}}", {"query": "best bars", ...})
        """
        template = text
        params = {}
        text_lower = text.lower()
        
        # Sort params by length (longest first) to avoid partial replacements
        sorted_params = sorted(
            self._param_value_to_name.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for value_lower, param_name in sorted_params:
            if value_lower in text_lower:
                # Find the original case version
                idx = text_lower.find(value_lower)
                original_value = text[idx:idx + len(value_lower)]
                
                # Replace in template
                placeholder = f"{{{{{param_name}}}}}"
                template = template[:idx] + placeholder + template[idx + len(value_lower):]
                text_lower = template.lower()
                
                params[param_name] = original_value
                self.logger.debug(f"  Template: '{original_value}' → {placeholder}")
        
        return template, params
    
    def convert_workflow_recipe_to_goals(
        self,
        recipe: WorkflowRecipe
    ) -> GoalWorkflow:
        """
        Convert an existing WorkflowRecipe to a GoalWorkflow.
        
        This allows upgrading legacy recipes to goal-based execution.
        """
        goal_steps = []
        
        for i, step in enumerate(recipe.steps):
            # Get next steps for outcome analysis
            next_steps = recipe.steps[i + 1: i + 3]
            
            goal = self._convert_recipe_step_to_goal(
                step=step,
                step_index=i,
                all_steps=recipe.steps,
                next_steps=next_steps
            )
            
            if goal:
                goal_steps.append(goal)
        
        # Build GoalWorkflow
        workflow = GoalWorkflow(
            workflow_id=recipe.workflow_id,
            name=recipe.name,
            description=recipe.description,
            parameters={
                name: param.example_value 
                for name, param in recipe.parameters.items()
            },
            steps=goal_steps,
            created_from_session=recipe.created_from_session,
            voice_analyzed=recipe.voice_analyzed,
            gemini_enriched=recipe.gemini_enriched
        )
        
        return workflow
    
    def _analyze_outcome(
        self, 
        step: SemanticStep, 
        next_steps: List[SemanticStep]
    ) -> Dict[str, Any]:
        """Analyze what happened AFTER this step."""
        outcome = {
            "url_changed": False,
            "new_url": None,
            "new_domain": None,
            "domain_changed": False,
            "page_type_changed": False,
            "new_page_type": None,
            "app_changed": False,
            "new_app": None,
            "data_extracted": False,
            "extracted_value": None,
        }
        
        current_domain = self._extract_domain(step.url_before)
        
        for next_step in next_steps:
            # URL change detection
            if next_step.url_before and step.url_before:
                if next_step.url_before != step.url_before:
                    outcome["url_changed"] = True
                    outcome["new_url"] = next_step.url_before
                    
                    next_domain = self._extract_domain(next_step.url_before)
                    outcome["new_domain"] = next_domain
                    
                    if next_domain and current_domain and next_domain != current_domain:
                        outcome["domain_changed"] = True
                    break
            
            # App change detection
            if next_step.app_name != step.app_name:
                outcome["app_changed"] = True
                outcome["new_app"] = next_step.app_name
                break
            
            # Extraction detection (copy events)
            if "copy" in next_step.keyboard_shortcuts:
                outcome["data_extracted"] = True
        
        return outcome
    
    def _infer_goal_from_step(
        self,
        step: SemanticStep,
        step_index: int,
        all_steps: List[SemanticStep],
        outcome: Dict[str, Any],
        voice_context: Optional[VoiceContext],
        extraction_schemas: Dict[str, Any],
        step_analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[GoalStep]:
        """Infer a goal from a semantic step and its outcome."""
        step_analysis = step_analysis or {}
        
        # === SEARCH GOAL ===
        if step.intent == "search" and step.typed_values:
            return self._create_search_goal(step, outcome)
        
        # === NAVIGATION GOAL (domain change) ===
        # Now uses LLM analysis to determine if navigation should be flexible
        if step.intent == "select" and outcome["domain_changed"]:
            return self._create_navigation_goal(
                step, outcome, step_index, all_steps, step_analysis
            )
        
        # === SELECT GOAL (same domain click) ===
        if step.intent == "select" and not outcome["domain_changed"]:
            return self._create_select_goal(step, outcome, step_index, all_steps)
        
        # === EXTRACT GOAL ===
        if step.intent == "extract" or "copy" in step.keyboard_shortcuts:
            return self._create_extract_goal(step, extraction_schemas)
        
        # === WRITE GOAL ===
        if step.intent == "write":
            return self._create_write_goal(step)
        
        # === SAVE GOAL ===
        if step.intent == "save":
            return self._create_save_goal(step)
        
        # === LAUNCH APP GOAL ===
        if step.intent == "launch_app" or outcome["app_changed"]:
            return self._create_launch_goal(step, outcome)
        
        # === FALLBACK: Generic select/click ===
        if step.clicked_elements:
            return self._create_generic_click_goal(step, outcome)
        
        return None
    
    def _create_search_goal(
        self, 
        step: SemanticStep, 
        outcome: Dict[str, Any]
    ) -> GoalStep:
        """Create a search goal with parameterized template."""
        original_query = " ".join(step.typed_values) if step.typed_values else ""
        
        # Create template from the query using detected parameters
        template, params = self._create_template_from_text(original_query)
        
        # If no params detected, use the original query as-is
        if not params:
            template = original_query
            params = {"query": original_query}
        
        self.logger.info(f"  Search template: '{template}'")
        self.logger.info(f"  Parameters: {params}")
        
        strategies = [
            # Google-specific - use template (will be substituted at execution)
            Strategy(
                name="google_search",
                priority=100,
                selector='textarea[name="q"], input[name="q"]',
                input_value=template,  # Template with placeholders
                submit_after=True,
                requires_url_pattern="google.com"
            ),
            # Generic search inputs
            Strategy(
                name="search_input",
                priority=80,
                selector='input[type="search"], [role="searchbox"], [role="combobox"]',
                input_value=template,
                submit_after=True
            ),
            # Gemini fallback
            Strategy(
                name="gemini_find_search",
                priority=50,
                visual_description="search input box or search field",
                input_value=template,
                submit_after=True
            ),
        ]
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.SEARCH,
            goal_description=f"Search for: {template[:50]}",
            platform=step.platform,
            app_name=step.app_name,
            source_url_pattern=self._url_to_pattern(step.url_before),
            success_criteria=SuccessCriteria(
                url_changed=True,
                url_contains="/search" if "google" in (step.url_before or "") else None,
            ),
            strategies=strategies,
            parameters=params,  # Store the example values for reference
            template=template,  # Store the template
            fallback_to_agent=True,
            agent_goal_prompt=f"Find the search box and search for: {template}",
            original_step_id=step.step_id
        )
    
    def _create_navigation_goal(
        self,
        step: SemanticStep,
        outcome: Dict[str, Any],
        step_index: int,
        all_steps: List[SemanticStep],
        step_analysis: Optional[Dict[str, Any]] = None
    ) -> GoalStep:
        """
        Create a navigation goal (click that leads to new domain).
        
        FUNDAMENTAL FIX: Uses LLM analysis to determine if navigation is:
        - specific_site: Must go to THIS domain (user expressed intent)
        - any_result: Any relevant search result is acceptable
        - first_result: Just click the first result
        """
        step_analysis = step_analysis or {}
        target_domain = outcome.get("new_domain", "")
        target_url = outcome.get("new_url", "")
        
        # Get navigation intent from LLM analysis
        navigation_intent = step_analysis.get("navigation_intent", "any_result")
        llm_target_site = step_analysis.get("target_site")
        reasoning = step_analysis.get("reasoning", "")
        
        self.logger.debug(f"Navigation intent: {navigation_intent} (target: {llm_target_site})")
        self.logger.debug(f"LLM reasoning: {reasoning}")
        
        # Determine ordinal position for search results
        ordinal = self._get_click_ordinal(step_index, all_steps)
        
        # Check if this navigation is part of a site-filtered search
        is_site_filtered = self._is_from_site_filtered_search(step_index, all_steps)
        
        strategies = []
        
        # Strategy priority depends on navigation intent
        if navigation_intent == "specific_site" and (llm_target_site or is_site_filtered):
            # User wants a SPECIFIC site - prioritize domain-targeted strategy
            target_for_strategy = llm_target_site or target_domain
            strategies.append(Strategy(
                name="gemini_target_domain",
                priority=80,
                visual_description=f"clickable link or search result that leads to {target_for_strategy}"
            ))
        elif navigation_intent in ["any_result", "first_result"]:
            # User just wants A result - prioritize ordinal/generic strategies
            strategies.append(Strategy(
                name="gemini_generic_result",
                priority=80,
                visual_description="clickable search result link in the main content area"
            ))
        
        # Always add ordinal strategy as backup
        ordinal_desc = f"{ordinal} search result link in the main content area"
        strategies.append(Strategy(
            name="gemini_ordinal",
            priority=60,
            visual_description=ordinal_desc
        ))
        
        # Generic search result fallback
        strategies.append(Strategy(
            name="gemini_any_result",
            priority=40,
            visual_description="any clickable search result link, not in header or sidebar"
        ))
        
        # Coordinate fallback (lowest priority - positions change)
        if step.clicked_elements:
            elem = step.clicked_elements[0]
            coords = elem.get("coordinates")
            if coords:
                strategies.append(Strategy(
                    name="coordinates",
                    priority=10,
                    coordinates=coords
                ))
        
        # =====================================================================
        # BUILD SUCCESS CRITERIA (FLEXIBLE based on intent)
        # =====================================================================
        
        if navigation_intent == "specific_site":
            if is_site_filtered:
                # Use template for dynamic site filtering
                success_criteria = SuccessCriteria(
                    url_contains="{{site_filter}}",
                    url_changed=True,
                )
                goal_desc = "Navigate to {{site_filter}} result"
                agent_prompt = "Click on a search result from the specified site"
            elif llm_target_site:
                # LLM identified specific target site
                success_criteria = SuccessCriteria(
                    url_contains=llm_target_site,
                    url_changed=True,
                )
                goal_desc = f"Navigate to {llm_target_site} result"
                agent_prompt = f"Click on a search result from {llm_target_site}"
            else:
                # Fallback to recorded domain
                success_criteria = SuccessCriteria(
                    url_contains=target_domain if target_domain else None,
                    url_changed=True,
                )
                goal_desc = f"Navigate to {target_domain or 'search result'}"
                agent_prompt = f"Click on a search result from {target_domain}"
        else:
            # FLEXIBLE navigation - just need URL to change
            # This is the KEY FIX: don't require specific domain when intent is "any_result"
            success_criteria = SuccessCriteria(
                url_changed=True,
                # Don't specify url_contains - any navigation is acceptable!
            )
            goal_desc = "Navigate to a search result"
            agent_prompt = "Click on any relevant search result link"
            
            self.logger.info(f"  Using FLEXIBLE navigation (any result acceptable)")
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.NAVIGATE,
            goal_description=goal_desc,
            platform=step.platform,
            app_name=step.app_name,
            source_url_pattern=self._url_to_pattern(step.url_before),
            success_criteria=success_criteria,
            strategies=sorted(strategies, key=lambda s: -s.priority),
            parameters={"site_filter": llm_target_site} if is_site_filtered else {},
            fallback_to_agent=True,
            agent_goal_prompt=agent_prompt,
            original_step_id=step.step_id,
            # Store the navigation intent for debugging/review
            metadata={"navigation_intent": navigation_intent, "reasoning": reasoning}
        )
    
    def _is_from_site_filtered_search(self, step_index: int, all_steps: List[SemanticStep]) -> bool:
        """
        Check if this navigation comes after a search that used site_filter.
        
        We detect this by looking at the search query - if it mentions a site name
        that we parameterized, this navigation should be dynamic too.
        """
        # Look back for the most recent search step
        for i in range(step_index - 1, -1, -1):
            prev_step = all_steps[i]
            if prev_step.intent == "search":
                # Check if the search query contains any of our site_filter params
                if prev_step.typed_values:
                    query = " ".join(prev_step.typed_values).lower()
                    # Common site names that indicate filtering
                    site_indicators = ["zomato", "yelp", "google maps", "tripadvisor", 
                                       "booking.com", "amazon", "flipkart", "swiggy"]
                    for site in site_indicators:
                        if site in query:
                            return True
                break
        return False
    
    def _create_select_goal(
        self,
        step: SemanticStep,
        outcome: Dict[str, Any],
        step_index: int,
        all_steps: List[SemanticStep]
    ) -> GoalStep:
        """Create a select/click goal (same domain)."""
        strategies = []
        
        # Determine if this is a "listing click" - clicking an item to see details
        is_listing_click = self._is_listing_click(step, step_index, all_steps)
        
        if is_listing_click:
            # For listing clicks, use scroll-aware strategy
            strategies.append(Strategy(
                name="gemini_click_listing",
                priority=80,
                visual_description="clickable listing card, item, or result that leads to a detail page. May need to scroll to find listings."
            ))
        
        # Gemini fallback
        ordinal = self._get_click_ordinal(step_index, all_steps)
        strategies.append(Strategy(
            name="gemini_visual",
            priority=50,
            visual_description=f"{ordinal} clickable element in the content area"
        ))
        
        # Coordinate fallback (low priority - positions change)
        if step.clicked_elements:
            elem = step.clicked_elements[0]
            coords = elem.get("coordinates")
            if coords:
                strategies.append(Strategy(
                    name="coordinates",
                    priority=10,
                    coordinates=coords
                ))
        
        # Success criteria
        if is_listing_click and outcome["url_changed"]:
            # For listing clicks, we MUST navigate to a new URL (detail page)
            criteria = SuccessCriteria(
                url_changed=True,
                timeout_success=False  # Don't accept timeout as success
            )
            goal_desc = "Click on a listing to view details"
            agent_prompt = "Find and click on a listing card or item to navigate to its detail page. Scroll down if needed to find listings."
        else:
            # For generic clicks, be more lenient
            criteria = SuccessCriteria(timeout_success=True)
            if outcome["url_changed"]:
                criteria.url_changed = True
            goal_desc = f"Click element in {step.app_name}"
            agent_prompt = "Click on the interactive element"
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.SELECT,
            goal_description=goal_desc,
            platform=step.platform,
            app_name=step.app_name,
            source_url_pattern=self._url_to_pattern(step.url_before),
            success_criteria=criteria,
            strategies=sorted(strategies, key=lambda s: -s.priority),
            fallback_to_agent=True,
            agent_goal_prompt=agent_prompt,
            original_step_id=step.step_id
        )
    
    def _is_listing_click(
        self, 
        step: SemanticStep, 
        step_index: int, 
        all_steps: List[SemanticStep]
    ) -> bool:
        """
        Detect if this click is on a listing item (e.g., restaurant card on Zomato).
        
        Indicators:
        - On a list/search results page
        - Followed by an extract step (user wants data from detail page)
        - URL contains list-like patterns
        """
        url = step.url_before or ""
        url_lower = url.lower()
        
        # Check if we're on a list/search page
        list_patterns = [
            "/search", "/results", "/restaurants", "/hotels", "/products",
            "/listings", "/places", "/bars", "/cafes", "/shops",
            "q=", "query=", "search?", "?s=", "/collection", "/category"
        ]
        on_list_page = any(p in url_lower for p in list_patterns)
        
        # Check if next step is extract (user wants to extract from detail)
        followed_by_extract = False
        if step_index + 1 < len(all_steps):
            next_step = all_steps[step_index + 1]
            if next_step.intent == "extract" or "copy" in next_step.keyboard_shortcuts:
                followed_by_extract = True
        
        # Also check 2 steps ahead (might be: click -> wait -> extract)
        if step_index + 2 < len(all_steps):
            next_next = all_steps[step_index + 2]
            if next_next.intent == "extract" or "copy" in next_next.keyboard_shortcuts:
                followed_by_extract = True
        
        return on_list_page or followed_by_extract
    
    def _create_extract_goal(
        self,
        step: SemanticStep,
        extraction_schemas: Dict[str, Any]
    ) -> GoalStep:
        """Create an extraction goal."""
        # Find matching extraction schema
        schema = None
        for key, s in extraction_schemas.items():
            # Match by timestamp or just use first available
            schema = s
            break
        
        schema_dict = {}
        if schema:
            if hasattr(schema, 'to_gemini_schema'):
                schema_dict = schema.to_gemini_schema()
            elif isinstance(schema, dict):
                schema_dict = schema
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.EXTRACT,
            goal_description="Extract data from current page",
            platform=step.platform,
            app_name=step.app_name,
            source_url_pattern=self._url_to_pattern(step.url_before),
            success_criteria=SuccessCriteria(
                min_extracted_count=1
            ),
            strategies=[
                Strategy(
                    name="gemini_vision_extract",
                    priority=100,
                    visual_description="Extract structured data from the page"
                )
            ],
            extraction_schema=schema_dict,
            fallback_to_agent=False,  # Extraction doesn't need agent loop
            original_step_id=step.step_id
        )
    
    def _create_write_goal(self, step: SemanticStep) -> GoalStep:
        """Create a write/type goal with template support."""
        original_text = " ".join(step.typed_values) if step.typed_values else ""
        
        # Check if this is a paste operation (uses extracted data)
        if "paste" in step.keyboard_shortcuts:
            return self._create_paste_goal(step)
        
        # Check if this is just a label (should be typed literally, not templated)
        if self._is_label_text(original_text):
            return GoalStep(
                step_id=f"goal_{step.step_id}",
                step_number=step.step_number,
                goal_type=GoalType.WRITE,
                goal_description=f"Type label: {original_text[:30]}",
                platform=step.platform,
                app_name=step.app_name,
                success_criteria=SuccessCriteria(timeout_success=True),
                strategies=[
                    Strategy(
                        name="focused_type",
                        priority=100,
                        input_value=original_text  # Type literally, no template
                    )
                ],
                parameters={},  # No parameters for labels
                template=None,  # No template
                fallback_to_agent=False,
                original_step_id=step.step_id
            )
        
        # Create template from text for parameterization
        template, params = self._create_template_from_text(original_text)
        
        # If template has placeholders, use it; otherwise use original
        if template == original_text:
            params = {"text": original_text}
        
        strategies = []
        
        # Desktop accessibility
        if step.platform == "desktop":
            if step.clicked_elements:
                elem = step.clicked_elements[0]
                if elem.get("accessibility_role"):
                    strategies.append(Strategy(
                        name="accessibility_type",
                        priority=90,
                        accessibility_role=elem.get("accessibility_role"),
                        accessibility_name=elem.get("accessibility_name"),
                        input_value=template
                    ))
            
            strategies.append(Strategy(
                name="focused_type",
                priority=70,
                input_value=template
            ))
            
            strategies.append(Strategy(
                name="gemini_find_input",
                priority=50,
                visual_description="text input field or text area",
                input_value=template
            ))
        
        # Browser
        else:
            if step.clicked_elements:
                elem = step.clicked_elements[0]
                if elem.get("selector"):
                    strategies.append(Strategy(
                        name="selector_type",
                        priority=100,
                        selector=elem["selector"],
                        input_value=template
                    ))
            
            strategies.append(Strategy(
                name="focused_type",
                priority=70,
                input_value=template
            ))
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.WRITE,
            goal_description=f"Type: {template[:30]}...",
            platform=step.platform,
            app_name=step.app_name,
            success_criteria=SuccessCriteria(timeout_success=True),
            strategies=sorted(strategies, key=lambda s: -s.priority),
            parameters=params,
            template=template,
            fallback_to_agent=True,
            agent_goal_prompt=f"Find a text input and type: {template[:50]}",
            original_step_id=step.step_id
        )
    
    def _create_paste_goal(self, step: SemanticStep) -> GoalStep:
        """Create a paste goal that uses extracted data."""
        # For paste operations, the content comes from extracted data
        # We create a template placeholder for the paste content
        
        strategies = [
            Strategy(
                name="paste_content",
                priority=100,
                shortcut_keys="command+v"
            )
        ]
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.SHORTCUT,
            goal_description="Paste extracted content",
            platform=step.platform,
            app_name=step.app_name,
            success_criteria=SuccessCriteria(timeout_success=True),
            strategies=strategies,
            parameters={},  # Content comes from extracted_data at runtime
            template="{{extracted_content}}",  # Filled from clipboard/extracted
            fallback_to_agent=False,
            original_step_id=step.step_id
        )
    
    def _create_save_goal(self, step: SemanticStep) -> GoalStep:
        """Create a save goal (Cmd+S or similar)."""
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.SAVE,
            goal_description="Save document",
            platform=step.platform,
            app_name=step.app_name,
            success_criteria=SuccessCriteria(timeout_success=True),
            strategies=[
                Strategy(
                    name="save_shortcut",
                    priority=100,
                    shortcut_keys="command+s"
                )
            ],
            original_step_id=step.step_id
        )
    
    def _create_launch_goal(
        self, 
        step: SemanticStep, 
        outcome: Dict[str, Any]
    ) -> GoalStep:
        """Create an app launch goal."""
        target_app = outcome.get("new_app") or step.app_name
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.LAUNCH,
            goal_description=f"Switch to {target_app}",
            platform="desktop",
            app_name=target_app,
            success_criteria=SuccessCriteria(
                app_active=target_app
            ),
            strategies=[
                Strategy(
                    name="activate_app",
                    priority=100
                ),
                Strategy(
                    name="launch_app",
                    priority=50
                )
            ],
            fallback_to_agent=False,
            original_step_id=step.step_id
        )
    
    def _extract_bundled_shortcuts(
        self,
        step: SemanticStep,
        prev_app: Optional[str]
    ) -> List[GoalStep]:
        """
        Extract paste/save/write goals from steps that have bundled shortcuts.
        
        This handles the case where paste/save happened in app A but got recorded
        as part of the "switch to app B" event.
        """
        bundled_goals = []
        
        # Check for paste shortcut
        if "paste" in step.keyboard_shortcuts:
            # Paste goal - use the PREVIOUS app (where paste actually happened)
            app_for_paste = prev_app or step.app_name
            bundled_goals.append(GoalStep(
                step_id=f"goal_{step.step_id}_paste",
                step_number=step.step_number,
                goal_type=GoalType.SHORTCUT,
                goal_description=f"Paste extracted content in {app_for_paste}",
                platform="desktop",
                app_name=app_for_paste,
                success_criteria=SuccessCriteria(timeout_success=True),
                strategies=[
                    Strategy(
                        name="paste_content",
                        priority=100,
                        shortcut_keys="command+v"
                    )
                ],
                template="{{extracted_content}}",
                fallback_to_agent=False,
                original_step_id=step.step_id
            ))
            self.logger.info(f"  Extracted bundled PASTE goal for {app_for_paste}")
        
        # Check for save shortcut
        if "save" in step.keyboard_shortcuts:
            app_for_save = prev_app or step.app_name
            bundled_goals.append(GoalStep(
                step_id=f"goal_{step.step_id}_save",
                step_number=step.step_number,
                goal_type=GoalType.SAVE,
                goal_description=f"Save document in {app_for_save}",
                platform="desktop",
                app_name=app_for_save,
                success_criteria=SuccessCriteria(timeout_success=True),
                strategies=[
                    Strategy(
                        name="save_shortcut",
                        priority=100,
                        shortcut_keys="command+s"
                    )
                ],
                original_step_id=step.step_id
            ))
            self.logger.info(f"  Extracted bundled SAVE goal for {app_for_save}")
        
        return bundled_goals
    
    def _create_generic_click_goal(
        self, 
        step: SemanticStep,
        outcome: Dict[str, Any]
    ) -> GoalStep:
        """Create a generic click goal as fallback."""
        strategies = []
        
        if step.clicked_elements:
            elem = step.clicked_elements[0]
            
            if elem.get("selector"):
                strategies.append(Strategy(
                    name="selector_click",
                    priority=100,
                    selector=elem["selector"]
                ))
            
            if elem.get("text"):
                strategies.append(Strategy(
                    name="text_click",
                    priority=80,
                    text_match=elem["text"][:50]
                ))
            
            coords = elem.get("coordinates")
            if coords:
                strategies.append(Strategy(
                    name="coordinates",
                    priority=20,
                    coordinates=coords
                ))
        
        strategies.append(Strategy(
            name="gemini_visual",
            priority=50,
            visual_description="interactive element to click"
        ))
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=GoalType.SELECT,
            goal_description=f"Click in {step.app_name}",
            platform=step.platform,
            app_name=step.app_name,
            success_criteria=SuccessCriteria(timeout_success=True),
            strategies=sorted(strategies, key=lambda s: -s.priority),
            fallback_to_agent=True,
            agent_goal_prompt="Click on the element",
            original_step_id=step.step_id
        )
    
    def _convert_recipe_step_to_goal(
        self,
        step: WorkflowStep,
        step_index: int,
        all_steps: List[WorkflowStep],
        next_steps: List[WorkflowStep]
    ) -> Optional[GoalStep]:
        """Convert a WorkflowStep to a GoalStep."""
        
        # Infer navigation target from next steps
        target_domain = None
        target_url = None
        for next_step in next_steps:
            if next_step.expected_url_pattern:
                target_domain = next_step.expected_url_pattern
                break
        
        # Map intent to goal type
        intent_to_goal = {
            "search": GoalType.SEARCH,
            "select": GoalType.SELECT,
            "navigate": GoalType.NAVIGATE,
            "write": GoalType.WRITE,
            "extract": GoalType.EXTRACT,
            "save": GoalType.SAVE,
            "launch_app": GoalType.LAUNCH,
        }
        
        goal_type = intent_to_goal.get(step.intent, GoalType.SELECT)
        
        # Build strategies from element_reference
        strategies = []
        ref = step.element_reference
        
        if ref:
            if ref.selector:
                strategies.append(Strategy(
                    name="selector",
                    priority=100,
                    selector=ref.selector
                ))
            
            if ref.text:
                strategies.append(Strategy(
                    name="text",
                    priority=80,
                    text_match=ref.text
                ))
            
            if ref.role:
                strategies.append(Strategy(
                    name="role",
                    priority=70,
                    role=ref.role
                ))
            
            if ref.visual_hint:
                strategies.append(Strategy(
                    name="gemini_visual",
                    priority=50,
                    visual_description=ref.visual_hint
                ))
            
            if ref.coordinates:
                strategies.append(Strategy(
                    name="coordinates",
                    priority=10,
                    coordinates=ref.coordinates
                ))
        
        # Build success criteria
        criteria = SuccessCriteria()
        
        if step.expected_url_pattern:
            criteria.url_contains = step.expected_url_pattern
        
        if step.completion_signal:
            if step.completion_signal.required_page_type:
                criteria.page_type = step.completion_signal.required_page_type
            if step.completion_signal.type == "url_change":
                criteria.url_changed = True
        
        if goal_type == GoalType.EXTRACT:
            criteria.min_extracted_count = 1
        
        if criteria.is_empty():
            criteria.timeout_success = True
        
        # Handle extraction schema
        extraction_schema = None
        if step.extraction_schema:
            extraction_schema = step.extraction_schema.to_gemini_schema() if hasattr(step.extraction_schema, 'to_gemini_schema') else {}
        
        return GoalStep(
            step_id=f"goal_{step.step_id}",
            step_number=step.step_number,
            goal_type=goal_type,
            goal_description=step.description,
            platform=step.platform,
            app_name=step.app_name,
            success_criteria=criteria,
            strategies=sorted(strategies, key=lambda s: -s.priority) if strategies else [],
            parameters=step.parameter_bindings,
            extraction_schema=extraction_schema,
            template=step.template,
            clipboard_content=step.clipboard_content,
            fallback_to_agent=True,
            agent_goal_prompt=f"Achieve: {step.description}",
            confidence=step.confidence,
            original_step_id=step.step_id
        )
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        if not url:
            return None
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return domain if domain else None
        except:
            return None
    
    def _url_to_pattern(self, url: str) -> Optional[str]:
        """Convert URL to a pattern."""
        if not url:
            return None
        domain = self._extract_domain(url)
        if not domain:
            return None
        try:
            parsed = urlparse(url)
            if parsed.path and parsed.path != "/":
                return f"{domain}{parsed.path}"
            return domain
        except:
            return domain
    
    def _get_click_ordinal(self, step_index: int, all_steps: List) -> str:
        """Determine ordinal position of this click in a sequence."""
        ordinals = ["first", "second", "third", "fourth", "fifth"]
        
        # Count consecutive select steps before this one
        count = 0
        for i in range(step_index - 1, -1, -1):
            step = all_steps[i]
            intent = step.intent if hasattr(step, 'intent') else None
            if intent == "select":
                count += 1
            else:
                break
        
        return ordinals[min(count, 4)]
    
    def _is_label_text(self, text: str) -> bool:
        """
        Check if text is a label/annotation (should be typed literally).
        
        GENERIC approach - detects labels by pattern:
        - Ends with colon (:)
        - Short text + separator
        - Common annotation patterns
        """
        import re
        stripped = text.strip()
        
        if len(stripped) <= 1:
            return False
        
        # Pattern 1: Ends with colon
        if stripped.endswith(':'):
            return True
        
        # Pattern 2: Word(s) + colon + optional space
        if re.match(r'^[\w\s]+\s*:\s*$', stripped):
            return True
        
        # Pattern 3: Short text with separator
        if len(stripped) < 30 and re.match(r'^[\w\s]+\s*[-|=:]\s*$', stripped):
            return True
        
        return False

