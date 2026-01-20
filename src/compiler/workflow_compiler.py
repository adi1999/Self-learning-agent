"""Workflow compiler - converts sessions into reusable recipes with Gemini enrichment."""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import re
import json

from src.models.session_artifact import SessionArtifact, VoiceTranscription
from src.models.semantic_trace import SemanticTrace, SemanticStep, VoiceContext
from src.models.workflow_recipe import (
    WorkflowRecipe, WorkflowStep, WorkflowParameter,
    ElementReference, CompletionSignal, FailurePolicy,
    ExtractionSchema, ExtractionField
)
from src.models.goal_step import GoalWorkflow, GoalStep
from src.interpreter.segmenter import TimelineSegmenter
from src.interpreter.intent_classifier import IntentClassifier
from src.interpreter.voice_analyzer import VoiceAnalyzer
from src.compiler.parameter_detector import ParameterDetector
from src.compiler.template_detector import TemplateDetector
from src.compiler.goal_inferrer import GoalInferrer
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client
from src.utils.gemini_client import gemini_client
from src.utils.config import config


class WorkflowCompiler:
    """
    Compiles session recordings into reusable workflow recipes.
    
    Enhanced Pipeline:
    1. Transcribe voice (moved from recording phase)
    2. Segment timeline into steps
    3. Classify step intents (GPT-4o)
    4. Analyze voice narration (GPT-4o)
    5. Detect parameters (GPT-4o)
    6. **NEW: Gemini visual enrichment for extraction steps**
    7. Detect templates
    8. Build workflow recipe
    """
    
    def __init__(self, use_llm: bool = True, use_gemini: bool = True):
        """
        Initialize compiler.
        
        Args:
            use_llm: Use GPT-4o for classification and parameter detection
            use_gemini: Use Gemini for visual enrichment of extraction steps
        """
        self.segmenter = TimelineSegmenter()
        self.intent_classifier = IntentClassifier(use_llm=use_llm)
        self.voice_analyzer = VoiceAnalyzer()
        self.parameter_detector = ParameterDetector()
        self.template_detector = TemplateDetector()
        self.goal_inferrer = GoalInferrer()  # NEW: Goal inference engine
        self.logger = setup_logger("WorkflowCompiler")
        
        self.use_llm = use_llm
        self.use_gemini = use_gemini and gemini_client.is_available
        
        if self.use_gemini:
            self.logger.info("Gemini enrichment enabled for extraction steps")
        else:
            self.logger.warning("Gemini not available - extraction schemas will be basic")
    
    def compile(
        self,
        session: SessionArtifact,
        workflow_name: str,
        description: Optional[str] = None
    ) -> WorkflowRecipe:
        """
        Compile session into workflow recipe.
        
        Args:
            session: Recorded session artifact
            workflow_name: Name for the workflow
            description: Optional description
        
        Returns:
            Compiled workflow recipe
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Compiling Workflow: {workflow_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Session: {session.session_id}")
        self.logger.info(f"Duration: {session.duration():.1f}s")
        self.logger.info(f"Timeline events: {len(session.timeline)}")
        
        # =====================================================================
        # Step 1: Transcribe voice (if available)
        # =====================================================================
        self.logger.info("\n[Step 1/8] Transcribing voice...")
        voice_transcription = None
        if session.voice_audio_path:
            voice_transcription = self._transcribe_voice(session)
            if voice_transcription:
                self.logger.info(f"  → Transcribed: {len(voice_transcription.text)} chars")
                self.logger.info(f"  → Preview: {voice_transcription.text[:100]}...")
        else:
            self.logger.info("  → No voice recording")
        
        # =====================================================================
        # Step 2: Segment timeline into steps
        # =====================================================================
        self.logger.info("\n[Step 2/8] Segmenting timeline...")
        semantic_steps = self.segmenter.segment(session)
        self.logger.info(f"  → {len(semantic_steps)} steps identified")
        
        # =====================================================================
        # Step 3: Classify step intents
        # =====================================================================
        self.logger.info("\n[Step 3/8] Classifying intents...")
        for step in semantic_steps:
            classification = self.intent_classifier.classify(step)
            step.intent = classification["intent"]
            step.confidence = classification["confidence"]
        
        # =====================================================================
        # Step 4: Analyze voice narration
        # =====================================================================
        self.logger.info("\n[Step 4/8] Analyzing voice narration...")
        voice_context = None
        if voice_transcription:
            voice_context = self.voice_analyzer.analyze(
                voice_transcription,
                semantic_steps
            )
            self.logger.info(f"  → Task goal: {voice_context.task_goal}")
            self.logger.info(f"  → {len(voice_context.extraction_hints)} extraction hints")
            self.logger.info(f"  → {len(voice_context.parameter_hints)} parameter hints")
        
        # =====================================================================
        # Step 5: Detect parameters
        # =====================================================================
        self.logger.info("\n[Step 5/8] Detecting parameters...")
        parameter_candidates = self.parameter_detector.detect(
            semantic_steps,
            voice_context
        )
        self.logger.info(f"  → {len(parameter_candidates)} parameters detected")
        for param in parameter_candidates:
            self.logger.info(f"     • {param.suggested_name}: '{param.value}' ({param.confidence:.2f})")
        
        # =====================================================================
        # Step 6: Gemini visual enrichment for extraction (NEW)
        # =====================================================================
        self.logger.info("\n[Step 6/8] Enriching extraction with Gemini...")
        extraction_schemas = self._enrich_extractions_with_gemini(
            session, semantic_steps, voice_context
        )
        self.logger.info(f"  → {len(extraction_schemas)} extraction schemas created")
        
        # =====================================================================
        # Step 7: Detect templates (BUT NOT FOR PASTE STEPS)
        # =====================================================================
        self.logger.info("\n[Step 7/8] Detecting templates...")
        extraction_hints = voice_context.extraction_hints if voice_context else []
        
        # Filter out steps that are paste operations - they shouldn't be templated
        non_paste_steps = [s for s in semantic_steps if not self._is_paste_step(s)]
        
        templates = self.template_detector.detect_templates_for_steps(
            steps=non_paste_steps,
            voice_context=voice_context,
            extraction_hints=extraction_hints
        )
        self.logger.info(f"  → {len(templates)} templates detected")
        
        # =====================================================================
        # Step 8: Build recipe
        # =====================================================================
        self.logger.info("\n[Step 8/8] Building workflow recipe...")
        recipe = self._build_recipe(
            session=session,
            workflow_name=workflow_name,
            description=description,
            semantic_steps=semantic_steps,
            parameter_candidates=parameter_candidates,
            voice_context=voice_context,
            templates=templates,
            extraction_schemas=extraction_schemas
        )
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Compilation Complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Workflow: {recipe.name}")
        self.logger.info(f"Steps: {len(recipe.steps)}")
        self.logger.info(f"Parameters: {len(recipe.parameters)}")
        self.logger.info(f"Extraction fields: {recipe.get_extraction_fields()}")
        
        if recipe.parameters:
            self.logger.info("\nParameters:")
            for name, param in recipe.parameters.items():
                self.logger.info(f"  • {name} ({param.param_type}): {param.example_value}")
        
        return recipe
    
    def compile_to_goals(
        self,
        session: SessionArtifact,
        workflow_name: str,
        description: Optional[str] = None
    ) -> GoalWorkflow:
        """
        Compile session into a GOAL-BASED workflow.
        
        This is the new, more robust compilation that creates goals
        with success criteria instead of action sequences.
        
        Args:
            session: Recorded session artifact
            workflow_name: Name for the workflow
            description: Optional description
        
        Returns:
            GoalWorkflow with goal-oriented steps
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Compiling GOAL Workflow: {workflow_name}")
        self.logger.info("=" * 60)
        
        # First, compile using traditional method to get semantic steps and parameters
        self.logger.info("\n[Phase 1] Standard compilation for parameters and extraction...")
        
        # Steps 1-5 from compile() - get semantic analysis
        voice_transcription = None
        if session.voice_audio_path:
            voice_transcription = self._transcribe_voice(session)
        
        semantic_steps = self.segmenter.segment(session)
        
        for step in semantic_steps:
            classification = self.intent_classifier.classify(step)
            step.intent = classification["intent"]
            step.confidence = classification["confidence"]
        
        voice_context = None
        if voice_transcription:
            voice_context = self.voice_analyzer.analyze(voice_transcription, semantic_steps)
        
        parameter_candidates = self.parameter_detector.detect(semantic_steps, voice_context)
        
        # Get extraction schemas
        extraction_schemas = self._enrich_extractions_with_gemini(
            session, semantic_steps, voice_context
        )
        
        # =====================================================================
        # Phase 2: Goal Inference (NEW)
        # =====================================================================
        self.logger.info("\n[Phase 2] Inferring goals from recorded actions...")
        
        # Enrich semantic steps with navigation outcomes from session
        self._enrich_steps_with_navigation_outcomes(semantic_steps, session)
        
        # Infer goals - pass detected parameters for template creation
        goal_steps = self.goal_inferrer.infer_goals_from_semantic_steps(
            semantic_steps=semantic_steps,
            voice_context=voice_context,
            extraction_schemas=extraction_schemas,
            detected_parameters=parameter_candidates  # Pass params for templating
        )
        
        self.logger.info(f"  → {len(goal_steps)} goals inferred")
        
        # Build parameters dict - ONLY include parameters actually USED in steps
        parameters = {}
        used_params = self._find_used_parameters(goal_steps)
        
        detected_but_unused = []
        for candidate in parameter_candidates:
            if candidate.confidence >= 0.5:
                if candidate.suggested_name in used_params:
                    parameters[candidate.suggested_name] = candidate.value
                else:
                    detected_but_unused.append(candidate.suggested_name)
        
        if detected_but_unused:
            self.logger.warning(f"  ⚠️  Detected but NOT USED in any step: {detected_but_unused}")
            self.logger.info(f"     (These were NOT added to the workflow)")
        
        self.logger.info(f"  → {len(parameters)} parameters will be included")
        
        # Create GoalWorkflow
        goal_workflow = GoalWorkflow(
            workflow_id=workflow_name.lower().replace(" ", "_"),
            name=workflow_name,
            description=description or f"Goal-based workflow from {session.session_id}",
            parameters=parameters,
            steps=goal_steps,
            created_from_session=session.session_id,
            voice_analyzed=bool(voice_context),
            gemini_enriched=bool(extraction_schemas)
        )
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Goal Workflow Compilation Complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Workflow: {goal_workflow.name}")
        self.logger.info(f"Goals: {len(goal_workflow.steps)}")
        self.logger.info(f"Parameters: {list(goal_workflow.parameters.keys())}")
        
        for i, goal in enumerate(goal_workflow.steps[:5]):  # Show first 5
            self.logger.info(f"  {i+1}. {goal.goal_type.value}: {goal.goal_description[:50]}")
        if len(goal_workflow.steps) > 5:
            self.logger.info(f"  ... and {len(goal_workflow.steps) - 5} more goals")
        
        return goal_workflow
    
    def _find_used_parameters(self, goal_steps: List) -> set:
        """
        Find which parameters are actually used in goal steps.
        
        Looks for {{param_name}} patterns in:
        - goal_description
        - template
        - strategy input_value
        - agent_goal_prompt
        """
        import re
        used = set()
        pattern = r"\{\{\s*(\w+)\s*\}\}"
        
        for step in goal_steps:
            # Check goal description
            if step.goal_description:
                matches = re.findall(pattern, step.goal_description)
                used.update(matches)
            
            # Check template
            if step.template:
                matches = re.findall(pattern, step.template)
                used.update(matches)
            
            # Check strategies
            for strat in step.strategies:
                if strat.input_value:
                    matches = re.findall(pattern, strat.input_value)
                    used.update(matches)
            
            # Check agent prompt
            if step.agent_goal_prompt:
                matches = re.findall(pattern, step.agent_goal_prompt)
                used.update(matches)
        
        return used
    
    def _enrich_steps_with_navigation_outcomes(
        self,
        steps: List[SemanticStep],
        session: SessionArtifact
    ):
        """
        Enrich semantic steps with navigation outcome data from the session.
        
        This adds url_after and domain information that helps the goal inferrer
        understand what the user was trying to achieve.
        """
        # Build a map of timestamp -> navigation_outcome
        nav_outcomes = {}
        for event in session.timeline:
            if event.navigation_outcome:
                nav_outcomes[event.timestamp] = event.navigation_outcome
        
        # Match steps to their outcomes
        for step in steps:
            # Find closest navigation outcome after step start
            step_time = step.start_timestamp
            closest_outcome = None
            closest_diff = float('inf')
            
            for ts, outcome in nav_outcomes.items():
                diff = ts - step_time
                if 0 <= diff < closest_diff:
                    closest_diff = diff
                    closest_outcome = outcome
            
            if closest_outcome and closest_diff < 2.0:  # Within 2 seconds
                # Add outcome data to step
                step.url_after = closest_outcome.url_after
                step.domain_after = closest_outcome.domain_after
                
                # Update url_before if not set
                if not step.url_before and closest_outcome.url_before:
                    step.url_before = closest_outcome.url_before
    
    def convert_recipe_to_goals(self, recipe: WorkflowRecipe) -> GoalWorkflow:
        """
        Convert an existing WorkflowRecipe to a GoalWorkflow.
        
        Useful for upgrading legacy recipes to the new goal-based format.
        """
        return self.goal_inferrer.convert_workflow_recipe_to_goals(recipe)
    
    def _is_paste_step(self, step: SemanticStep) -> bool:
        """Check if step is a paste operation (should not be templated)."""
        return "paste" in step.keyboard_shortcuts
    
    def _is_label_text(self, text: str) -> bool:
        """
        Check if text is just a label/annotation (ends with : or similar).
        Labels should be typed literally, not templated.
        """
        stripped = text.strip()
        # Labels typically end with colon, are short, have no dynamic content
        if stripped.endswith(':') and len(stripped) < 50:
            return True
        # Match pattern like "Name : " or "Rating:" or "Address : "
        if re.match(r'^[\w\s]+\s*:\s*$', stripped):
            return True
        return False
    
    def _transcribe_voice(self, session: SessionArtifact) -> Optional[VoiceTranscription]:
        """
        Transcribe voice recording using OpenAI's API (gpt-4o-transcribe).
        
        This is done at compile time instead of recording time.
        """
        if not session.voice_audio_path:
            return None
        
        session_dir = config.sessions_dir / session.session_id
        voice_path = session_dir / session.voice_audio_path
        
        if not voice_path.exists():
            self.logger.warning(f"Voice file not found: {voice_path}")
            return None
        
        if not llm_client.is_available:
            self.logger.warning("LLM client not available for transcription")
            return None
        
        self.logger.info(f"  Transcribing: {voice_path}")
        
        # Use OpenAI API via llm_client
        result = llm_client.transcribe_audio(voice_path)
        
        if not result:
            self.logger.warning("Transcription returned no result")
            return None
        
        return VoiceTranscription(
            text=result.get("text", ""),
            segments=result.get("segments", []),
            language=result.get("language"),
            duration=result.get("duration")
        )
    
    def _enrich_extractions_with_gemini(
        self,
        session: SessionArtifact,
        steps: List[SemanticStep],
        voice_context: Optional[VoiceContext]
    ) -> Dict[str, ExtractionSchema]:
        """Use Gemini to analyze pages where user performed copy operations."""
        if not self.use_gemini:
            return {}
        
        extraction_schemas = {}
        session_dir = config.sessions_dir / session.session_id
        
        # Find copy events with screenshots
        copy_events = []
        for event in session.timeline:
            for inp in event.input_events:
                if inp.shortcut == "copy" and inp.clipboard_content:
                    copy_events.append({
                        "timestamp": event.timestamp,
                        "clipboard": inp.clipboard_content,
                        "screenshot": event.screenshot_path
                    })
        
        if not copy_events:
            return {}
        
        voice_hints = voice_context.extraction_hints if voice_context else []
        
        for i, copy_event in enumerate(copy_events):
            screenshot_path = copy_event.get("screenshot")
            clipboard_content = copy_event.get("clipboard", "")
            
            if not screenshot_path:
                self.logger.warning(f"  Copy event {i+1} has no screenshot, skipping")
                continue
            
            full_screenshot_path = session_dir / screenshot_path
            if not full_screenshot_path.exists():
                self.logger.warning(f"  Screenshot not found: {full_screenshot_path}")
                continue
            
            self.logger.info(f"  Analyzing copy event {i+1}: '{clipboard_content[:30]}...'")
            
            analysis = gemini_client.analyze_extraction_page(
                screenshot_path=full_screenshot_path,
                copied_value=clipboard_content,
                voice_hints=voice_hints
            )
            
            if not analysis:
                self.logger.warning(f"  Gemini analysis failed for copy event {i+1}")
                continue
            
            all_fields = analysis.get("all_fields", {})
            
            schema = ExtractionSchema(
                page_type=analysis.get("page_type"),
                page_source=analysis.get("page_source"),
                layout_hints=analysis.get("layout_hints")
            )
            
            for field_name, field_info in all_fields.items():
                description = field_info.get("description", field_name)
                
                # Refine constraints to avoid generic list headers
                # If field looks like a name/title, add constraint
                if "name" in field_name.lower() or "title" in field_name.lower():
                    description += ". Specific entity name, NOT page title or list header (e.g. NOT 'Best Bars in...')."
                
                schema.fields[field_name] = ExtractionField(
                    description=description,
                    visual_hint=field_info.get("visual_hint"),
                    example_value=field_info.get("example_value"),
                    required=True
                )
            
            self.logger.info(f"    → Found {len(schema.fields)} extractable fields")
            
            timestamp_key = f"copy_{copy_event.get('timestamp', i)}"
            extraction_schemas[timestamp_key] = schema
        
        return extraction_schemas
    
    def _build_clipboard_map(self, session: SessionArtifact) -> Dict[float, str]:
        """
        Build a map of timestamp → clipboard content.
        Tracks the clipboard STATE at each point in time.
        """
        current_clipboard: Optional[str] = None
        clipboard_state: Dict[float, str] = {}
        
        for event in session.timeline:
            for inp in event.input_events:
                if inp.shortcut == "copy" and inp.clipboard_content:
                    current_clipboard = inp.clipboard_content
            
            if current_clipboard:
                clipboard_state[event.timestamp] = current_clipboard
        
        return clipboard_state

    def _create_paste_template(
        self, 
        clipboard_content: str, 
        extraction_schemas: List[ExtractionSchema]
    ) -> Optional[str]:
        """
        Match clipboard content to extraction fields and return a template.
        
        If the clipboard contains "Gatsby" and we have an extraction field
        "restaurant_name" with example_value "Gatsby", return "{{restaurant_name}}"
        """
        if not clipboard_content or not extraction_schemas:
            return None
        
        clipboard_lower = clipboard_content.strip().lower()
        
        for schema in extraction_schemas:
            if not schema or not schema.fields:
                continue
                
            for field_name, field_info in schema.fields.items():
                example_value = field_info.example_value
                if example_value:
                    # Check if clipboard matches the example value
                    if example_value.strip().lower() == clipboard_lower:
                        return f"{{{{{field_name}}}}}"
                    # Check if clipboard contains the example (for multi-field pastes)
                    if example_value.strip().lower() in clipboard_lower:
                        # Partial match - could be part of composite paste
                        pass
        
        return None

    def _get_clipboard_at_time(
        self, 
        clipboard_map: Dict[float, str], 
        timestamp: float
    ) -> Optional[str]:
        """Get clipboard content at a specific timestamp."""
        result = None
        for ts in sorted(clipboard_map.keys()):
            if ts <= timestamp:
                result = clipboard_map[ts]
            else:
                break
        return result

    def _build_recipe(
        self,
        session: SessionArtifact,
        workflow_name: str,
        description: Optional[str],
        semantic_steps: List[SemanticStep],
        parameter_candidates: list,
        voice_context: Optional[VoiceContext],
        templates: Dict[str, str],
        extraction_schemas: Dict[str, ExtractionSchema]
    ) -> WorkflowRecipe:
        """Build workflow recipe from compiled data."""
        
        # Create parameters
        parameters = {}
        param_value_to_name = {}  # Map values to parameter names
        
        for candidate in parameter_candidates:
            if candidate.confidence >= 0.5:
                param = WorkflowParameter(
                    name=candidate.suggested_name,
                    param_type=candidate.param_type,
                    description=f"Detected from: '{candidate.value}'",
                    example_value=candidate.value,
                    required=True
                )
                parameters[candidate.suggested_name] = param
                param_value_to_name[candidate.value] = candidate.suggested_name
        
        # Extract clipboard contents from session's copy events
        clipboard_contents = self._extract_clipboard_contents(session)
        self.logger.info(f"Found {len(clipboard_contents)} clipboard captures")
        
        extraction_index = 0
        extraction_schema_list = list(extraction_schemas.values())

        clipboard_map = self._build_clipboard_map(session)
        self.logger.info(f"Built clipboard map with {len(clipboard_map)} states")    
        
        workflow_steps = []
        
        for i, sem_step in enumerate(semantic_steps):
            # Check if this is a PASTE step - handle specially
            is_paste_step = "paste" in sem_step.keyboard_shortcuts
            
            # Determine action type
            action_type = self._determine_action_type(sem_step)
            
            # Build element reference
            element_ref = self._build_element_reference(sem_step, i, semantic_steps)
            
            # Get clipboard content for paste steps
            clipboard_for_step = None
            shortcut = None
            template_for_paste = None  # NEW: template for paste steps
            
            if sem_step.keyboard_shortcuts:
                shortcut = sem_step.keyboard_shortcuts[0]
            
            if shortcut == "paste":
                clipboard_for_step = self._get_clipboard_at_time(
                    clipboard_map, 
                    sem_step.start_timestamp
                )
                if clipboard_for_step:
                    preview = clipboard_for_step[:40] + "..." if len(clipboard_for_step) > 40 else clipboard_for_step
                    self.logger.debug(f"Paste at t={sem_step.start_timestamp:.1f}s: '{preview}'")
                    
                    # NEW: Try to match clipboard content to extraction fields
                    # and create a template placeholder
                    template_for_paste = self._create_paste_template(
                        clipboard_for_step, 
                        extraction_schema_list
                    )
                    if template_for_paste:
                        self.logger.info(f"  Paste step gets template: {template_for_paste}")
            
            # Build parameter bindings - use smart template building
            param_bindings = self._build_parameter_bindings(
                sem_step, param_value_to_name, parameter_candidates
            )
            
            # Get extraction schema for extract steps
            extraction_schema = None
            if sem_step.intent == "extract":
                if extraction_index < len(extraction_schema_list):
                    extraction_schema = extraction_schema_list[extraction_index]
                    extraction_index += 1
                elif voice_context and voice_context.extraction_hints:
                    extraction_schema = ExtractionSchema()
                    for hint in voice_context.extraction_hints:
                        field_name = hint.lower().replace(" ", "_")
                        extraction_schema.fields[field_name] = ExtractionField(
                            description=f"Extract {hint}"
                        )
            
            # Get template - BUT NOT for paste steps or label-only text
            template = None
            if not is_paste_step:
                template = templates.get(sem_step.step_id)
                
                # Also skip template if it's just label text
                if sem_step.typed_values:
                    typed_text = " ".join(sem_step.typed_values)
                    if self._is_label_text(typed_text):
                        template = None
            
            # Determine completion signal
            completion = self._determine_completion_signal(sem_step)
            
            # Detect expected navigation target from URL changes
            expected_url_pattern = None
            if sem_step.url_before and sem_step.url_after:
                # Extract domains
                before_domain = self._extract_domain(sem_step.url_before)
                after_domain = self._extract_domain(sem_step.url_after)
                
                # If domain changed, capture the target domain
                if before_domain != after_domain and after_domain:
                    expected_url_pattern = after_domain
                    self.logger.debug(f"  Step {i+1} navigated: {before_domain} → {after_domain}")
            
            workflow_step = WorkflowStep(
                step_id=sem_step.step_id,
                step_number=sem_step.step_number,
                intent=sem_step.intent,
                description=sem_step.description if hasattr(sem_step, 'description') else f"{sem_step.intent} in {sem_step.app_name}",
                platform=sem_step.platform,
                app_name=sem_step.app_name,
                action_type=action_type,
                element_reference=element_ref,
                parameter_bindings=param_bindings,
                extraction_schema=extraction_schema,
                template=template if not is_paste_step else template_for_paste,  # CHANGED: use paste template
                shortcut=shortcut,
                clipboard_content=clipboard_for_step if not template_for_paste else None,  # CHANGED: clear if templated
                completion_signal=completion,
                confidence=sem_step.confidence,
                screenshot_path=sem_step.screenshot_paths[0] if sem_step.screenshot_paths else None,
                expected_url_pattern=expected_url_pattern  # NEW: navigation target
            )
            
            workflow_steps.append(workflow_step)
        
        # Post-processing: Link extraction requirements to previous navigation steps
        # If Step N is extract(page_type="restaurant_detail"), Step N-1 (click) must ensure we land there.
        for i in range(1, len(workflow_steps)):
            current_step = workflow_steps[i]
            prev_step = workflow_steps[i-1]
            
            # If current step is extraction with page type requirement
            if (current_step.intent == "extract" and 
                current_step.extraction_schema and 
                current_step.extraction_schema.page_type):
                
                req_type = current_step.extraction_schema.page_type
                
                # And previous step is a click/navigate on browser
                if (prev_step.platform == "browser" and 
                    prev_step.action_type in ["click", "navigate"]):
                    
                    # Add requirement to previous step's completion signal
                    if not prev_step.completion_signal:
                        prev_step.completion_signal = CompletionSignal(type="url_change")
                    
                    prev_step.completion_signal.required_page_type = req_type
                    self.logger.info(f"  Linked Step {prev_step.step_number} navigation -> Step {current_step.step_number} extraction ({req_type})")
        
        recipe = WorkflowRecipe(
            workflow_id=workflow_name.lower().replace(" ", "_"),
            name=workflow_name,
            description=description or f"Workflow from session {session.session_id}",
            parameters=parameters,
            steps=workflow_steps,
            failure_policy=FailurePolicy(use_gemini_fallback=self.use_gemini),
            created_from_session=session.session_id,
            voice_analyzed=bool(voice_context),
            gemini_enriched=bool(extraction_schemas)
        )
        
        return recipe
    
    def _extract_clipboard_contents(self, session: SessionArtifact) -> List[str]:
        """Extract all clipboard contents from copy events in chronological order."""
        contents = []
        for event in session.timeline:
            for inp in event.input_events:
                if inp.shortcut == "copy" and inp.clipboard_content:
                    contents.append(inp.clipboard_content)
        return contents
    
    def _determine_action_type(self, step: SemanticStep) -> str:
        """Determine action type from step."""
        intent_to_action = {
            "search": "type",
            "select": "click",
            "navigate": "click",
            "write": "type",
            "extract": "extract",
            "save": "shortcut",
            "launch_app": "launch_app",
        }
        
        action = intent_to_action.get(step.intent, "click")
        
        # Override based on actual actions - BUT NOT FOR EXTRACT INTENT
        # Extract steps should stay as "extract" even if they have copy shortcut
        if step.intent == "extract":
            return "extract"  # Always return extract for extract intent
        
        if step.keyboard_shortcuts:
            if any(s in step.keyboard_shortcuts for s in ["save", "copy", "paste"]):
                action = "shortcut"
        elif step.typed_values:
            action = "type"
        elif step.clicked_elements and not step.typed_values:
            action = "click"
        
        return action
    
    def _build_element_reference(
        self, 
        step: SemanticStep, 
        step_index: int = 0, 
        all_steps: Optional[List[SemanticStep]] = None
    ) -> Optional[ElementReference]:
        """Build element reference with semantic positioning."""
        
        if not step.clicked_elements:
            return None
        
        clicked = step.clicked_elements[0]
        text = clicked.get("text", "")
        
        # Check if we should store the text or generalize
        should_store_text = self._should_store_element_text(step, text)
        
        # Generate semantic hint based on workflow context
        # Pass the app_name for desktop-aware hints
        semantic_hint = self._generate_semantic_hint(step, step_index, all_steps or [])
        
        # For desktop apps, ALWAYS store text if available (it's essential for finding elements)
        if step.platform == "desktop" and text and not should_store_text:
            # Store it anyway for desktop - we need every hint we can get
            should_store_text = len(text) <= 100  # Allow longer text for desktop
        
        # Extract bbox from clicked element
        bbox = clicked.get("bbox")
        if bbox and not isinstance(bbox, list):
            bbox = None
        
        # Extract absolute_position
        abs_pos = clicked.get("absolute_position")
        
        return ElementReference(
            selector=clicked.get("selector"),
            role=clicked.get("role"),
            accessibility_role=clicked.get("accessibility_role"),
            accessibility_name=clicked.get("accessibility_name"),
            text=text if should_store_text else None,
            coordinates=clicked.get("coordinates"),
            bbox=bbox,
            absolute_position=abs_pos,
            visual_hint=semantic_hint,
        )

    def _should_store_element_text(self, step: SemanticStep, text: str) -> bool:
        """
        Determine if element text should be stored or generalized.
        
        Don't store text that's clearly result-specific.
        """
        if not text:
            return False
        
        # Don't store addresses
        if self._looks_like_address(text):
            return False
        
        # Don't store very long text (likely page-specific content)
        if len(text) > 50:
            return False
        
        # Don't store if step is selecting on a results page
        if step.intent == "select" and self._is_on_results_page(step):
            return False
        
        # Store short, generic text (like button labels)
        return True

    def _looks_like_address(self, text: str) -> bool:
        """Check if text looks like an address."""
        import re
        text_lower = text.lower()
        address_patterns = [
            r'\d+[,/]',  # Numbers followed by comma or slash
            r'\d+\s+\w+\s+(street|st|road|rd|avenue|ave|lane|ln)',
            r'opposite|near|behind',
        ]
        return any(re.search(p, text_lower) for p in address_patterns)

    def _is_on_results_page(self, step: SemanticStep) -> bool:
        """Check if this step is on a search results page."""
        url = step.url_before or ""
        return any(x in url.lower() for x in [
            '/search', 'q=', 'query=', '/results', 
            'zomato.com', 'yelp.com', 'google.com/search'
        ])

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL (e.g., 'zomato.com' from 'https://www.zomato.com/delhi/...')."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove 'www.' prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain if domain else None
        except Exception:
            return None

    def _generate_semantic_hint(
        self, 
        step: SemanticStep, 
        step_index: int, 
        all_steps: List[SemanticStep]
    ) -> str:
        """Generate semantic description of what to click."""
        
        coords = None
        clicked_element = None
        if step.clicked_elements:
            clicked_element = step.clicked_elements[0]
            coords = clicked_element.get("coordinates")
        
        # =========================================================================
        # DESKTOP APPS: Provide rich, actionable hints
        # =========================================================================
        if step.platform == "desktop":
            parts = []
            
            # Include app name for context
            app_name = step.app_name
            
            # Get accessibility role if available
            if clicked_element:
                a11y_role = clicked_element.get("accessibility_role")
                a11y_name = clicked_element.get("accessibility_name")
                text = clicked_element.get("text", "")
                
                if a11y_role:
                    # Map macOS accessibility roles to human-readable descriptions
                    role_map = {
                        "AXTextArea": "text area",
                        "AXTextField": "text field",
                        "AXButton": "button",
                        "AXLink": "link",
                        "AXStaticText": "text",
                        "AXImage": "image",
                        "AXGroup": "group",
                        "AXCheckBox": "checkbox",
                        "AXTable": "table",
                        "AXList": "list",
                        "AXRow": "row",
                        "AXCell": "cell",
                    }
                    readable_role = role_map.get(a11y_role, a11y_role.replace("AX", "").lower())
                    parts.append(readable_role)
                
                # Include name if meaningful
                if a11y_name and len(a11y_name) < 50:
                    parts.append(f'named "{a11y_name}"')
                elif text and len(text) < 50:
                    # Use text content if name not available
                    preview = text[:40] + "..." if len(text) > 40 else text
                    parts.append(f'containing "{preview}"')
            
            # Add app context
            parts.append(f"in {app_name}")
            
            if parts:
                return " ".join(parts)
            
            # Fallback for desktop
            return f"interactive element in {app_name}"
        
        # =========================================================================
        # BROWSER: Existing logic for search results pages
        # =========================================================================
        # Count consecutive select steps (not just same URL)
        # Look backwards until we hit a non-select step
        select_count = 0
        for i in range(step_index - 1, -1, -1):
            prev = all_steps[i]
            if prev.intent == "select":
                select_count += 1
            else:
                break  # Stop counting when we hit extract/write/navigate/etc
        
        is_results_page = self._is_on_results_page(step)
        
        if step.intent == "select" and is_results_page:
            ordinals = ["first", "second", "third", "fourth", "fifth"]
            ordinal = ordinals[min(select_count, 4)]
            
            if coords and len(coords) >= 2:
                x = coords[0]
                if x < 600:
                    return f"{ordinal} search result in the main content area"
                else:
                    return f"result item on the right sidebar"
            
            return f"{ordinal} search result link"
        
        # Position-based hint for non-results pages
        if coords and len(coords) >= 2:
            x, y = coords[0], coords[1]
            h_pos = "left" if x < 400 else "right" if x > 900 else "center"
            v_pos = "top" if y < 250 else "bottom" if y > 550 else "middle"
            return f"clickable element in the {v_pos} {h_pos} area"
        
        # Fallback based on intent
        intent_hints = {
            "search": "search input field",
            "select": "clickable link or button",
            "navigate": "navigation link",
            "write": "text input field",
            "extract": "content area",
        }
        return intent_hints.get(step.intent, "interactive element")
    
    def _build_parameter_bindings(
        self,
        step: SemanticStep,
        param_value_to_name: Dict[str, str],
        all_params: list
    ) -> Dict[str, str]:
        """
        Build parameter bindings with SMART template construction.
        
        PRIORITY: If we have voice-derived component params, use LLM to build
        a composite template instead of using a single full-text param.
        """
        bindings = {}
        
        for typed_value in step.typed_values:
            # Collect all params that might be relevant to this typed value
            exact_substring_params = []  # Params whose values are substrings
            full_text_param = None       # Param that matches entire text
            voice_component_params = []  # Voice-derived params that might be components
            
            for param_value, param_name in param_value_to_name.items():
                param_lower = param_value.lower().strip()
                typed_lower = typed_value.lower().strip()
                
                if param_lower == typed_lower:
                    # Exact full-text match
                    full_text_param = (param_value, param_name)
                elif param_lower in typed_lower:
                    # Substring match
                    exact_substring_params.append((param_value, param_name))
            
            # Also collect voice-derived params that aren't exact substrings
            # but might be semantic components (LLM will figure out the mapping)
            for param in all_params:
                if param.was_in_voice and param.suggested_name not in [p[1] for p in exact_substring_params]:
                    # Don't include the full-text param if it came from voice
                    if full_text_param and param.suggested_name == full_text_param[1]:
                        continue
                    voice_component_params.append((param.value, param.suggested_name))
            
            # Decision logic
            all_component_params = exact_substring_params + voice_component_params
            
            if len(exact_substring_params) >= 2:
                # Multiple exact substrings → build template
                template = self._build_smart_template(typed_value, exact_substring_params)
                bindings["value"] = template
                self.logger.info(f"  Smart template (exact): '{typed_value}' → '{template}'")
            
            elif len(all_component_params) >= 2:
                # We have voice-derived components → let LLM figure out the template
                template = self._build_smart_template_with_voice_params(
                    typed_value, 
                    all_component_params,
                    full_text_param
                )
                bindings["value"] = template
                self.logger.info(f"  Smart template (voice): '{typed_value}' → '{template}'")
            
            elif len(exact_substring_params) == 1:
                # Single substring match
                param_value, param_name = exact_substring_params[0]
                template = self._replace_preserving_case(
                    typed_value, param_value, f"{{{{{param_name}}}}}"
                )
                bindings["value"] = template
            
            elif full_text_param:
                # Only full-text match, no components
                bindings["value"] = f"{{{{{full_text_param[1]}}}}}"
            
            else:
                # No match, use literal
                if "value" not in bindings:
                    bindings["value"] = typed_value
        
        return bindings


    def _build_smart_template_with_voice_params(
        self,
        typed_value: str,
        component_params: List[Tuple[str, str]],
        full_text_param: Optional[Tuple[str, str]]
    ) -> str:
        """
        Use LLM to build a template when we have voice-derived component params.
        
        The LLM figures out:
        1. How the component values map to parts of the typed text
        2. What connecting words/structure to preserve
        3. How to combine everything into a template
        """
        if not llm_client.is_available:
            # Fallback: just use full text param if available
            if full_text_param:
                return f"{{{{{full_text_param[1]}}}}}"
            return typed_value
        
        params_desc = "\n".join([
            f'  - {name}: "{value}"' for value, name in component_params
        ])
        
        prompt = f"""Convert this typed text into a template using the given parameters.

        Typed text: "{typed_value}"

        Available parameters (from user's voice narration):
        {params_desc}

        Your task:
        1. Figure out how each parameter value relates to parts of the typed text
        2. Replace those parts with {{{{param_name}}}} placeholders
        3. Keep connecting words (in, at, for, near, on, etc.) as static text
        4. If a parameter doesn't appear in the text but is relevant, append it naturally
        
        CRITICAL RULES:
        - Do NOT duplicate words. If "best" is part of the {{query}} parameter, do NOT write "best {{{{query}}}}".
        - If a parameter value fully contains a word from the text, replace successfuly.
        - Example: Text="best bars in delhi", Query="best bars" -> "{{{{query}}}} in delhi" (NOT "best {{{{query}}}} in delhi")

        Examples:
        - Text: "best sushi restaurants in san francisco"
        Params: query="sushi restaurants", location="san francisco"  
        Result: "best {{{{query}}}} in {{{{location}}}}"

        - Text: "best bars in bangalore"
        Params: query="best bars", location="bangalore"
        Result: "{{{{query}}}} in {{{{location}}}}"  <-- Notice "best" is NOT repeated because it's in the query param

        - Text: "cheap flights to paris"
        Params: destination="paris", origin="new york", site="kayak"
        Result: "cheap flights to {{{{destination}}}} from {{{{origin}}}} {{{{site}}}}"

        IMPORTANT: 
        - Don't use special search syntax like "site:" - just append naturally
        - Make it read like a natural search query

        Return ONLY the template string, nothing else."""

        try:
            result = llm_client.complete(prompt, temperature=0.0, max_tokens=200)
            if result:
                result = result.strip().strip('"').strip("'")
                # Validate it has at least one placeholder
                if "{{" in result and "}}" in result:
                    return result
        except Exception as e:
            self.logger.debug(f"LLM template building failed: {e}")
        
        # Fallback
        if full_text_param:
            return f"{{{{{full_text_param[1]}}}}}"
        return typed_value
    
    def _replace_preserving_case(self, text: str, old: str, new: str) -> str:
        """Replace substring preserving surrounding text."""
        # Case-insensitive find and replace
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        return pattern.sub(new, text, count=1)
    
    def _build_smart_template(
        self, 
        typed_value: str, 
        params: List[Tuple[str, str]]
    ) -> str:
        """
        Use LLM to build a template from typed value and detected params.
        
        Example:
            typed_value: "best bar in bangalore"
            params: [("best bar", "query"), ("bangalore", "location")]
            result: "{{query}} in {{location}}"
        """
        if not llm_client.is_available:
            # Fallback: simple sequential replacement
            result = typed_value
            for param_value, param_name in params:
                result = self._replace_preserving_case(
                    result, param_value, f"{{{{{param_name}}}}}"
                )
            return result
        
        params_desc = "\n".join([
            f'  - "{pv}" should become {{{{{{pn}}}}}}' 
            for pv, pn in params
        ])
        
        prompt = f"""Convert this typed text into a template with parameter placeholders.

Original text: "{typed_value}"

Parameters to replace:
{params_desc}

Rules:
1. Replace each parameter value with its {{{{param_name}}}} placeholder
2. Keep connecting words (in, at, for, near, etc.) as static text
3. Preserve spaces and punctuation
4. The template should work when different values are substituted

Example:
- Original: "best sushi in san francisco"
- Parameters: "best sushi" → {{{{query}}}}, "san francisco" → {{{{location}}}}
- Template: "{{{{query}}}} in {{{{location}}}}"

Return ONLY the template string, nothing else."""
        
        try:
            result = llm_client.complete(prompt, temperature=0.0, max_tokens=200)
            if result:
                # Clean up response
                result = result.strip().strip('"').strip("'")
                # Validate it contains the expected placeholders
                for _, param_name in params:
                    if f"{{{{{param_name}}}}}" not in result:
                        self.logger.warning(f"Template missing placeholder {{{{{param_name}}}}}, using fallback")
                        raise ValueError("Missing placeholder")
                return result
        except Exception as e:
            self.logger.debug(f"LLM template building failed: {e}")
        
        # Fallback: simple replacement
        result = typed_value
        for param_value, param_name in params:
            result = self._replace_preserving_case(
                result, param_value, f"{{{{{param_name}}}}}"
            )
        return result
    
    def _determine_completion_signal(self, step: SemanticStep) -> Optional[CompletionSignal]:
        """Determine how to detect step completion."""
        if step.intent == "search":
            return CompletionSignal(
                type="url_change",
                timeout_ms=10000
            )
        
        if step.intent == "navigate":
            return CompletionSignal(
                type="network_idle",
                timeout_ms=10000
            )
        
        if step.intent == "select":
            return CompletionSignal(
                type="content_change",
                timeout_ms=5000
            )
        
        return CompletionSignal(
            type="timeout",
            timeout_ms=2000
        )
    
    def _create_template_from_typed(
        self, 
        write_step: SemanticStep, 
        extraction_schema: ExtractionSchema
    ) -> Optional[str]:
        """Create a template from typed text by matching against extraction fields."""
        if not write_step.typed_values:
            return None
        
        written_text = "\n".join(write_step.typed_values)
        template = written_text
        
        for field_name, field_info in extraction_schema.fields.items():
            example_value = field_info.example_value
            if example_value and example_value in template:
                placeholder = f"{{{{{field_name}}}}}"
                template = template.replace(example_value, placeholder)
        
        if "{{" in template:
            return template
        
        return None