"""Voice narration analysis for workflow understanding."""
from typing import List, Optional
from src.models.session_artifact import VoiceTranscription
from src.models.semantic_trace import SemanticStep, VoiceContext
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client
import json


class VoiceAnalyzer:
    """
    Analyzes voice narration to extract actionable context.
    
    Extracts:
    - Field labels ("this is the restaurant name")
    - Parameter hints ("searching for sushi")
    - Task connections (weather → jacket search)
    - Extraction hints (what data to extract)
    """
    
    def __init__(self):
        self.logger = setup_logger("VoiceAnalyzer")
    
    def analyze(
        self,
        transcription: VoiceTranscription,
        steps: List[SemanticStep]
    ) -> Optional[VoiceContext]:
        """
        Analyze voice transcription in context of workflow steps.
        
        Args:
            transcription: Voice transcription with segments
            steps: Semantic steps for context
        
        Returns:
            VoiceContext with extracted information
        """
        if not transcription or not transcription.text:
            self.logger.debug("No voice transcription to analyze")
            return None
        
        if not llm_client.is_available:
            self.logger.warning("LLM not available for voice analysis")
            return self._basic_analysis(transcription)
        
        return self._llm_analysis(transcription, steps)
    
    def _basic_analysis(self, transcription: VoiceTranscription) -> VoiceContext:
        """Basic keyword-based analysis without LLM."""
        context = VoiceContext()
        text_lower = transcription.text.lower()
        
        # Look for common field label patterns
        field_patterns = [
            ("restaurant name", "restaurant_name"),
            ("the name", "name"),
            ("rating", "rating"),
            ("address", "address"),
            ("phone number", "phone"),
            ("price", "price"),
            ("email", "email"),
        ]
        
        for phrase, field in field_patterns:
            if phrase in text_lower:
                context.field_labels.append({
                    "phrase": phrase,
                    "field_name": field
                })
                context.extraction_hints.append(field)
        
        # Look for search-related keywords
        search_keywords = ["searching for", "looking for", "find", "search"]
        for keyword in search_keywords:
            if keyword in text_lower:
                # Try to extract what they're searching for
                idx = text_lower.find(keyword)
                after = transcription.text[idx + len(keyword):idx + len(keyword) + 50]
                words = after.strip().split()[:3]
                if words:
                    context.parameter_hints.append({
                        "value": " ".join(words),
                        "type": "search_query"
                    })
        
        return context
    
    def _llm_analysis(
        self,
        transcription: VoiceTranscription,
        steps: List[SemanticStep]
    ) -> Optional[VoiceContext]:
        """Analyze voice using LLM for deeper understanding."""
        
        # Format steps for context
        steps_summary = []
        for step in steps:
            summary = {
                "step": step.step_number,
                "intent": step.intent,
                "app": step.app_name,
                "typed": step.typed_values[:2] if step.typed_values else None,
                "time": f"{step.start_timestamp:.1f}s"
            }
            steps_summary.append(summary)
        
        prompt = f"""Analyze this voice narration from a workflow recording.

## Voice Transcript
"{transcription.text}"

## Workflow Steps Performed
{json.dumps(steps_summary, indent=2)}

## Task
Extract actionable information from the voice narration:

1. **Field Labels**: When user explicitly names data they're looking at or extracting
   - Example: "this is the restaurant name" → field_name: "restaurant_name"
   - Example: "getting the rating" → field_name: "rating"

2. **Parameter Hints**: Values mentioned that would change on re-run
   - Example: "searching for sushi" → value: "sushi", type: "cuisine"
   - Example: "in San Francisco" → value: "San Francisco", type: "location"

3. **Task Connections**: When user explains how one search relates to another
   - Example: "weather is cold so searching warm jackets" → weather connects to jacket search

4. **Extraction Hints**: What data fields should be extracted from pages
   - Example: "getting restaurant info" → ["restaurant_name", "rating", "address"]

5. **Task Goal**: The overall purpose of the workflow

Return JSON:
{{
    "field_labels": [
        {{"phrase": "exact phrase used", "field_name": "normalized_field_name", "timestamp_approx": 0.0}}
    ],
    "parameter_hints": [
        {{"value": "the value", "type": "cuisine|location|query|name|date|other", "timestamp_approx": 0.0}}
    ],
    "task_connections": [
        {{"source": "what was looked up", "target": "what it affects", "relationship": "description"}}
    ],
    "extraction_hints": ["field1", "field2"],
    "task_goal": "brief description of what user is trying to accomplish"
}}

If no relevant information for a category, use empty array/null.
"""
        
        try:
            result = llm_client.complete_json(prompt)
            
            if result:
                context = VoiceContext(
                    field_labels=result.get("field_labels", []),
                    parameter_hints=result.get("parameter_hints", []),
                    task_connections=result.get("task_connections", []),
                    extraction_hints=result.get("extraction_hints", []),
                    task_goal=result.get("task_goal")
                )
                
                self.logger.info(f"Voice analysis complete: {len(context.parameter_hints)} params, "
                               f"{len(context.extraction_hints)} extraction hints")
                
                return context
        
        except Exception as e:
            self.logger.error(f"Voice analysis failed: {e}")
        
        return self._basic_analysis(transcription)
    
    def get_voice_for_step(
        self,
        step: SemanticStep,
        transcription: VoiceTranscription
    ) -> Optional[str]:
        """
        Get voice transcript segment that corresponds to a step.
        
        Uses timestamp alignment to find relevant voice.
        """
        if not transcription or not transcription.segments:
            return None
        
        relevant_text = []
        
        for segment in transcription.segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            
            # Check if segment overlaps with step
            if (seg_start <= step.end_timestamp and seg_end >= step.start_timestamp):
                relevant_text.append(segment.get("text", ""))
        
        if relevant_text:
            return " ".join(relevant_text).strip()
        
        return None
    
    def enrich_steps_with_voice(
        self,
        steps: List[SemanticStep],
        transcription: VoiceTranscription
    ) -> List[SemanticStep]:
        """Add voice transcript segments to steps."""
        if not transcription:
            return steps
        
        for step in steps:
            voice_segment = self.get_voice_for_step(step, transcription)
            if voice_segment:
                step.voice_transcript = voice_segment
        
        return steps