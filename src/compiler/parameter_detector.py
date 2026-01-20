"""LLM-based parameter detection for workflow generalization."""
from typing import List, Dict, Any, Optional, Set
from src.models.semantic_trace import SemanticStep, ParameterCandidate, VoiceContext
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client
import json
import re


# Known site/platform names for voice hint detection
KNOWN_SITES = {
    "zomato", "yelp", "google maps", "tripadvisor", "swiggy", 
    "uber eats", "doordash", "grubhub", "opentable", "booking.com",
    "airbnb", "expedia", "kayak", "amazon", "ebay", "walmart",
    "target", "best buy", "linkedin", "indeed", "glassdoor"
}


class ParameterDetector:
    """
    Detects which values should become parameters.
    
    Uses multiple signals:
    - Typed text in search contexts → high parameter likelihood
    - Values mentioned in voice narration → creates parameters even if not typed
    - Site/platform mentions in voice → optional site_filter parameter
    - Numeric values with operations → likely parameters
    """
    
    def __init__(self):
        self.logger = setup_logger("ParameterDetector")
        self._used_names: Set[str] = set()
    
    def detect(
        self,
        steps: List[SemanticStep],
        voice_context: Optional[VoiceContext] = None
    ) -> List[ParameterCandidate]:
        """Detect parameter candidates from steps and voice."""
        self._used_names = set()
        candidates = []
        
        # Source 1: Typed values with context
        typed_contexts = self._collect_typed_contexts(steps)
        
        if typed_contexts:
            self.logger.info(f"Analyzing {len(typed_contexts)} typed values with LLM")
            for ctx in typed_contexts:
                self.logger.debug(f"  Value: '{ctx['text']}' (intent: {ctx['intent']}, app: {ctx['app']})")
        
        # Source 2: Voice-mentioned parameters
        voice_hints = []
        if voice_context and voice_context.parameter_hints:
            voice_hints = voice_context.parameter_hints
            self.logger.info(f"Voice hints: {len(voice_hints)} potential parameters")
            for hint in voice_hints:
                self.logger.debug(f"  Voice param: '{hint.get('value')}' (type: {hint.get('type')})")
        
        # Detect from typed values
        if typed_contexts:
            if llm_client.is_available:
                typed_candidates = self._detect_with_llm(typed_contexts, voice_hints)
            else:
                typed_candidates = self._detect_with_heuristics(typed_contexts, voice_hints)
            candidates.extend(typed_candidates)
        
        # Create parameters from voice hints (including site mentions)
        voice_candidates = self._create_from_voice_hints(voice_hints, typed_contexts, steps)
        candidates.extend(voice_candidates)
        
        # NEW: Detect site/platform mentions in voice transcription
        if voice_context and voice_context.task_goal:
            site_candidates = self._detect_site_from_voice(voice_context)
            candidates.extend(site_candidates)
        
        # Deduplicate by VALUE, keeping highest confidence
        candidates = self._deduplicate_by_value(candidates)
        
        # Remove parameters that are actively composed of other parameters
        candidates = self._remove_consumed_parameters(candidates)
        
        # Ensure all names are unique
        candidates = self._ensure_unique_names(candidates)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        self.logger.info(f"Detected {len(candidates)} parameter candidates")
        for c in candidates:
            self.logger.info(f"  {c.suggested_name}: '{c.value}' (conf: {c.confidence:.2f})")
        
        return candidates
    
    def _collect_typed_contexts(self, steps: List[SemanticStep]) -> List[Dict]:
        """Collect typed values with their context."""
        contexts = []
        
        for step in steps:
            for typed_value in step.typed_values:
                if not typed_value or len(typed_value.strip()) == 0:
                    continue
                
                # Skip labels/annotations (they're structural, not data)
                if self._is_label_text(typed_value):
                    self.logger.debug(f"  Skipping label text: '{typed_value}'")
                    continue
                
                contexts.append({
                    "text": typed_value,
                    "step_id": step.step_id,
                    "app": step.app_name,
                    "platform": step.platform,
                    "intent": step.intent,
                    "url": step.url_before,
                    "was_submitted": step.boundary_reason == "submit"
                })
        
        return contexts
    
    def _is_label_text(self, text: str) -> bool:
        """
        Check if text is a label/annotation (should NOT be a parameter).
        
        GENERIC approach - detects labels by pattern, not specific content:
        - Ends with colon (:)
        - Short text followed by separator
        - Common annotation patterns
        """
        stripped = text.strip()
        
        # Empty or very short
        if len(stripped) <= 1:
            return False
        
        # Pattern 1: Ends with colon (e.g., "Name:", "Address :", "Rating:")
        if stripped.endswith(':'):
            return True
        
        # Pattern 2: Word(s) followed by colon and space (e.g., "Name : ", "Rating : ")
        if re.match(r'^[\w\s]+\s*:\s*$', stripped):
            return True
        
        # Pattern 3: Short text with separator at end (e.g., "Price -", "Date |")
        if len(stripped) < 30 and re.match(r'^[\w\s]+\s*[-|=:]\s*$', stripped):
            return True
        
        # Pattern 4: Looks like a label followed by placeholder text
        if re.match(r'^[\w\s]+:\s*_+\s*$', stripped):  # e.g., "Name: ____"
            return True
        
        return False
    
    def _detect_site_from_voice(
        self, 
        voice_context: VoiceContext
    ) -> List[ParameterCandidate]:
        """
        Detect site/platform mentions in voice for optional site_filter parameter.
        
        Example: If user says "I want to find bars on Zomato", detect "zomato"
        as a site_filter parameter that could be appended to search queries.
        """
        candidates = []
        
        # Check task_goal and parameter_hints for site mentions
        text_to_check = voice_context.task_goal or ""
        
        for hint in (voice_context.parameter_hints or []):
            value = str(hint.get("value", "")).lower()
            hint_type = str(hint.get("type", "")).lower()
            
            # Check if this is a site mention
            if value in KNOWN_SITES or hint_type in ["site", "platform", "website"]:
                # Check if we already have this as a parameter
                candidate = ParameterCandidate(
                    value=value,
                    suggested_name="site_filter",
                    param_type="string",
                    confidence=0.65,
                    was_in_voice=True,
                    semantic_type="site_filter"
                )
                candidates.append(candidate)
                self.logger.info(f"  Detected site filter from voice: '{value}'")
        
        # Also scan task_goal for known sites
        task_goal_lower = text_to_check.lower()
        for site in KNOWN_SITES:
            if site in task_goal_lower:
                # Check if already added
                if not any(c.value.lower() == site for c in candidates):
                    candidate = ParameterCandidate(
                        value=site,
                        suggested_name="site_filter",
                        param_type="string",
                        confidence=0.6,
                        was_in_voice=True,
                        semantic_type="site_filter"
                    )
                    candidates.append(candidate)
                    self.logger.info(f"  Detected site filter from task goal: '{site}'")
        
        return candidates
    
    def _detect_with_llm(
        self,
        typed_contexts: List[Dict],
        voice_hints: List[Dict]
    ) -> List[ParameterCandidate]:
        """Detect parameters using LLM with SEMANTIC CHUNKING."""
        
        prompt = f"""Analyze these typed values and identify SEMANTIC COMPONENTS that should be PARAMETERS.

## Typed Values with Context
{json.dumps(typed_contexts, indent=2)}

## Voice Hints (user mentioned these as important values)
{json.dumps(voice_hints, indent=2)}

## CRITICAL: Semantic Chunking Rules

For search queries like "best pizza places in bangalore zomato":
1. **query**: The main search intent ("best pizza places")
2. **location**: Geographic reference ("bangalore")  
3. **site_filter**: Specific website/platform ("zomato")

IMPORTANT RULES:
- Extract COMPLETE WORDS only - NEVER fragment words
  - WRONG: "aces in b" (fragment of "places in bangalore")
  - RIGHT: "bangalore" (complete word)
  
- Identify semantic boundaries at connecting words (in, at, on, for, near, from, to)
  - "best bars in delhi" → query="best bars", location="delhi"
  - "flights from NYC to LA" → origin="NYC", destination="LA"

- Recognize common site/platform names as site_filter:
  zomato, yelp, google maps, tripadvisor, swiggy, uber eats, amazon, flipkart, booking.com, airbnb, etc.

- The FULL query can ALSO be a parameter, but component params are preferred if they exist

## Return JSON array:
[
    {{
        "value": "exact value as typed (complete words only)",
        "is_parameter": true,
        "type": "query|location|site_filter|category|date|price|name|number|etc",
        "suggested_name": "unique_snake_case_name",
        "confidence": 0.9
    }}
]

## Examples:

Input: "best pizza places in bangalore zomato"
Output:
[
    {{"value": "best pizza places", "is_parameter": true, "type": "query", "suggested_name": "query", "confidence": 0.95}},
    {{"value": "bangalore", "is_parameter": true, "type": "location", "suggested_name": "location", "confidence": 0.95}},
    {{"value": "zomato", "is_parameter": true, "type": "site_filter", "suggested_name": "site_filter", "confidence": 0.9}}
]

Input: "cheap hotels near eiffel tower"
Output:
[
    {{"value": "cheap hotels", "is_parameter": true, "type": "query", "suggested_name": "query", "confidence": 0.9}},
    {{"value": "eiffel tower", "is_parameter": true, "type": "landmark", "suggested_name": "location", "confidence": 0.9}}
]

NEVER OUTPUT FRAGMENTS LIKE "aces in b" or "lore z" - these are INVALID.
"""
        
        try:
            result = llm_client.complete_json(prompt)
            
            if not result:
                self.logger.warning("LLM returned empty result, falling back to heuristics")
                return self._detect_with_heuristics(typed_contexts, voice_hints)
            
            self.logger.debug(f"LLM response type: {type(result)}, content: {result}")
            
            # Handle various response formats
            if isinstance(result, dict):
                # Try common keys
                result = result.get("parameters", result.get("results", result.get("items", [])))
            
            if not isinstance(result, list):
                self.logger.warning(f"LLM response not a list: {type(result)}, falling back to heuristics")
                return self._detect_with_heuristics(typed_contexts, voice_hints)
            
            if len(result) == 0:
                self.logger.warning("LLM returned empty list, falling back to heuristics")
                return self._detect_with_heuristics(typed_contexts, voice_hints)
            
            candidates = []
            for item in result:
                # Accept both is_parameter=true and items without that field (assume true)
                is_param = item.get("is_parameter", True)
                if is_param:
                    value = str(item.get("value", ""))
                    if not value:
                        continue
                    
                    # =========================================================
                    # VALIDATION: Reject fragmented/invalid parameters
                    # =========================================================
                    if not self._is_valid_parameter_value(value, typed_contexts):
                        self.logger.warning(f"  Rejecting invalid param fragment: '{value}'")
                        continue
                    
                    source_steps = [
                        ctx["step_id"] for ctx in typed_contexts
                        if ctx["text"] == value or value in ctx["text"]
                    ]
                    
                    was_in_search = any(
                        ctx["intent"] == "search" or ctx["was_submitted"]
                        for ctx in typed_contexts
                        if ctx["text"] == value or value in ctx["text"]
                    )
                    
                    candidate = ParameterCandidate(
                        value=value,
                        suggested_name=item.get("suggested_name", item.get("name", "param")),
                        param_type="number" if self._is_numeric(value) else "string",
                        confidence=item.get("confidence", 0.7),
                        source_step_ids=source_steps,
                        was_in_search=was_in_search,
                        semantic_type=item.get("type")
                    )
                    candidates.append(candidate)
            
            # If LLM found nothing but we have typed values, use heuristics
            if len(candidates) == 0 and len(typed_contexts) > 0:
                self.logger.warning("LLM found no params but we have typed values, using heuristics")
                return self._detect_with_heuristics(typed_contexts, voice_hints)
            
            return candidates
        
        except Exception as e:
            self.logger.error(f"LLM parameter detection failed: {e}")
            import traceback
            traceback.print_exc()
            return self._detect_with_heuristics(typed_contexts, voice_hints)
    
    def _detect_with_heuristics(
        self,
        typed_contexts: List[Dict],
        voice_hints: List[Dict]
    ) -> List[ParameterCandidate]:
        """Fallback heuristic-based parameter detection - more aggressive."""
        candidates = []
        voice_values = {str(h.get("value", "")).lower() for h in voice_hints}
        
        self.logger.info("Using heuristic parameter detection")
        
        for ctx in typed_contexts:
            value = ctx["text"]
            confidence = 0.0
            semantic_type = None
            
            # SEARCH INTENT = ALWAYS A PARAMETER
            if ctx["intent"] == "search":
                confidence = 0.85  # High confidence for search
                semantic_type = "search_query"
                self.logger.debug(f"  Search query detected: '{value}'")
            
            # SUBMITTED = VERY LIKELY A PARAMETER
            elif ctx["was_submitted"]:
                confidence = 0.75
                semantic_type = "input"
                self.logger.debug(f"  Submitted value detected: '{value}'")
            
            # Browser input
            if ctx["platform"] == "browser":
                confidence += 0.15
            
            # Voice mentioned
            for voice_val in voice_values:
                if voice_val in value.lower():
                    confidence += 0.2
                    self.logger.debug(f"  Voice match for: '{value}'")
                    break
            
            # Short values are more likely parameters (not long content)
            if len(value) < 100:
                confidence += 0.1
            
            # Contains specific patterns
            if re.search(r'\d', value):  # Has numbers
                confidence += 0.1
            
            # THRESHOLD: 0.3 is enough (was 0.4)
            if confidence >= 0.3:
                name = self._suggest_name(value, semantic_type)
                
                candidate = ParameterCandidate(
                    value=value,
                    suggested_name=name,
                    param_type="string",
                    confidence=min(confidence, 0.95),
                    source_step_ids=[ctx["step_id"]],
                    was_in_search=ctx["intent"] == "search",
                    semantic_type=semantic_type
                )
                candidates.append(candidate)
                self.logger.info(f"  Heuristic param: '{value[:30]}...' (conf: {confidence:.2f})")
        
        return candidates
    
    def _suggest_name(self, value: str, semantic_type: Optional[str] = None) -> str:
        """Suggest a parameter name based on value and context."""
        if semantic_type:
            return semantic_type.lower().replace(" ", "_")
        
        if self._is_numeric(value):
            return "number"
        
        # Use slugified value prefix
        slug = re.sub(r'[^a-z0-9]+', '_', value.lower())[:15].strip('_')
        return slug if slug else "param"
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric."""
        try:
            float(str(value).replace(',', ''))
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_valid_parameter_value(self, value: str, typed_contexts: List[Dict]) -> bool:
        """
        Validate that a parameter value is not a fragment.
        
        GENERIC approach - detects fragments by:
        1. Checking if the value is at a word boundary in the original text
        2. Rejecting values that cut words in half
        3. Rejecting very short non-numeric values
        """
        value = value.strip()
        
        # Empty or whitespace-only is invalid
        if not value:
            return False
        
        # Single character (except numbers) is usually invalid
        if len(value) == 1 and not value.isdigit():
            return False
        
        # Very short values (2-3 chars) that aren't numbers or common abbreviations
        if len(value) <= 3 and not self._is_numeric(value):
            # Common abbreviations are OK
            common_abbrevs = {"nyc", "la", "sf", "uk", "usa", "etc"}
            if value.lower() not in common_abbrevs:
                return False
        
        # Check if value appears at word boundaries in original text
        for ctx in typed_contexts:
            text = ctx.get("text", "").lower()
            value_lower = value.lower()
            
            if value_lower in text:
                # Find the position
                idx = text.find(value_lower)
                if idx >= 0:
                    # Check left boundary (should be start or preceded by space/punct)
                    left_ok = (idx == 0) or (not text[idx - 1].isalnum())
                    
                    # Check right boundary (should be end or followed by space/punct)
                    end_idx = idx + len(value_lower)
                    right_ok = (end_idx >= len(text)) or (not text[end_idx].isalnum())
                    
                    if left_ok and right_ok:
                        return True  # Value is at word boundary
                    else:
                        # This is a fragment cutting through a word
                        self.logger.debug(f"  Fragment detected: '{value}' in '{text}'")
                        return False
        
        # If value wasn't found in any context, it might be from voice hints
        # Be more lenient with those
        return len(value) >= 3
    
    def _create_from_voice_hints(
        self,
        voice_hints: List[Dict],
        typed_contexts: List[Dict],
        steps: List[SemanticStep]
    ) -> List[ParameterCandidate]:
        """Create parameter candidates from voice hints."""
        candidates = []
        all_typed_text = " ".join(ctx["text"] for ctx in typed_contexts)
        number_count = 0
        
        for hint in voice_hints:
            value = hint.get("value", "")
            hint_type = hint.get("type", "")
            
            if not value:
                continue
            
            # Skip results/outputs
            if hint_type and any(word in hint_type.lower() for word in ["result", "output", "answer"]):
                self.logger.debug(f"Skipping result value: {value}")
                continue
            
            # Skip site mentions (handled separately)
            if str(value).lower() in KNOWN_SITES:
                continue
            
            value_in_typed = str(value) in all_typed_text
            confidence = 0.75 if value_in_typed else 0.6
            
            if hint_type and hint_type.lower() not in ["number", "value", "param"]:
                suggested_name = hint_type.lower().replace(" ", "_")
            elif self._is_numeric(value):
                number_count += 1
                suggested_name = f"number_{number_count}"
            else:
                suggested_name = self._suggest_name(str(value))
            
            candidate = ParameterCandidate(
                value=str(value),
                suggested_name=suggested_name,
                param_type="number" if self._is_numeric(value) else "string",
                confidence=confidence,
                was_in_voice=True,
                semantic_type=hint_type if hint_type else None
            )
            candidates.append(candidate)
        
        return candidates
    
    def _ensure_unique_names(self, candidates: List[ParameterCandidate]) -> List[ParameterCandidate]:
        """Ensure all parameter names are unique."""
        used_names: Set[str] = set()
        result = []
        
        for c in candidates:
            original_name = c.suggested_name
            unique_name = original_name
            counter = 1
            
            while unique_name in used_names:
                counter += 1
                unique_name = f"{original_name}_{counter}"
            
            c.suggested_name = unique_name
            used_names.add(unique_name)
            result.append(c)
        
        return result
    
    def _deduplicate_by_value(self, candidates: List[ParameterCandidate]) -> List[ParameterCandidate]:
        """Remove duplicate candidates by VALUE, keeping highest confidence."""
        seen_values: Dict[str, ParameterCandidate] = {}
        
        for c in candidates:
            key = str(c.value).lower().strip()
            if key not in seen_values or c.confidence > seen_values[key].confidence:
                seen_values[key] = c
        
        return list(seen_values.values())

    def _remove_consumed_parameters(self, candidates: List[ParameterCandidate]) -> List[ParameterCandidate]:
        """
        Remove parameters that are fully 'consumed' by other more specific parameters.
        
        Example: 
        If we have:
        - search_text: "best bars in delhi" (from typing)
        - query: "best bars" (from voice)
        - location: "delhi" (from voice)
        
        We should remove 'search_text' so the compiler is forced to use the 
        composition of 'query' and 'location'.
        """
        if len(candidates) < 2:
            return candidates
            
        final_list = []
        # Sort by length descending (so we process containers first)
        sorted_candidates = sorted(candidates, key=lambda c: len(str(c.value)), reverse=True)
        # We need to map back to original objects to maintain order/integrity if needed, 
        # but here we're building a new list.
        
        # We want to check against ALL candidates, not just final_list
        all_candidates_map = {id(c): c for c in candidates}
        
        for container in sorted_candidates:
            container_val = str(container.value).lower().strip()
            if not container_val:
                continue
                
            contained_by = []
            
            for other in candidates:
                if other == container:
                    continue
                
                other_val = str(other.value).lower().strip()
                if not other_val:
                    continue
                    
                # Check if other is a substring
                if other_val in container_val:
                    contained_by.append(other)
            
            # Logic: If it's a search/typed string and we have components that match parts of it,
            # we consider it "decomposed" and remove the original full string.
            # We require at least one component, and if the container is very long, maybe more coverage?
            # For now, simplistic: if we have components and the container was a search/typed string (not voice),
            # prefer the components.
            
            should_remove = False
            if len(contained_by) > 0:
                # If container is FROM SEARCH/TYPING and components are FROM VOICE
                if container.was_in_search and any(c.was_in_voice for c in contained_by):
                    should_remove = True
                # Or if we have multiple components covering it
                elif len(contained_by) >= 2:
                    should_remove = True
            
            if should_remove:
                names = [c.suggested_name for c in contained_by]
                self.logger.info(f"  Removing redundant param '{container.suggested_name}' (covered by {names})")
                continue
                
            final_list.append(container)
            
        return final_list