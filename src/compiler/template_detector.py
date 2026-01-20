"""LLM-based template detection for write steps."""
from typing import Dict, List, Optional
from src.models.semantic_trace import SemanticStep, VoiceContext
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client
import json


class TemplateDetector:
    """
    Detects templates in written text using LLM inference.
    
    Instead of trying to match exact values (which we don't have at compile time),
    we use GPT-4o to infer the template structure from:
    - The typed text pattern
    - Voice narration hints
    - Extraction hints from earlier steps
    """
    
    def __init__(self):
        self.logger = setup_logger("TemplateDetector")
    
    def detect_template_llm(
        self,
        typed_text: str,
        voice_context: Optional[VoiceContext],
        extraction_hints: List[str]
    ) -> Optional[str]:
        """
        Use LLM to infer template from typed text.
        
        Args:
            typed_text: What the user typed (e.g., "Restaurant: Sushi Ran\nRating: 4.5")
            voice_context: Voice narration context
            extraction_hints: Fields that were/will be extracted
        
        Returns:
            Template string with {{placeholders}} or None
        """
        if not typed_text or len(typed_text) < 5:
            return None
        
        if not llm_client.is_available:
            return self._detect_with_heuristics(typed_text, extraction_hints)
        
        voice_info = ""
        if voice_context:
            if voice_context.extraction_hints:
                voice_info += f"\nVoice mentioned these fields: {voice_context.extraction_hints}"
            if voice_context.task_goal:
                voice_info += f"\nTask goal: {voice_context.task_goal}"
        
        prompt = f"""Analyze this text that a user typed into a notes/document app.
The user previously extracted data from a webpage and is now writing a summary.

## Typed Text
```
{typed_text}
```

## Known Extraction Fields
These fields were extracted from the source page: {extraction_hints}
{voice_info}

## Task
Convert the typed text into a TEMPLATE by replacing specific values with placeholders.

Rules:
1. Replace specific data values (names, numbers, addresses) with {{{{field_name}}}} placeholders
2. Keep structural text (labels like "Restaurant:", "Rating:", etc.) as-is
3. Use snake_case for placeholder names
4. Match placeholder names to the extraction fields when possible
5. If a value looks like extracted data but doesn't match known fields, infer an appropriate field name

## Examples
Input: "Restaurant: Sushi Ran\nRating: 4.5 stars\nAddress: 123 Main St"
Output: "Restaurant: {{{{restaurant_name}}}}\nRating: {{{{rating}}}} stars\nAddress: {{{{address}}}}"

Input: "Best bar found: Helen's Place (4.3 rating, 5337 reviews)"
Output: "Best bar found: {{{{name}}}} ({{{{rating}}}} rating, {{{{num_reviews}}}} reviews)"

Return JSON:
{{
    "template": "the template with placeholders",
    "fields_used": ["field1", "field2"],
    "confidence": 0.0-1.0
}}

If the text doesn't appear to contain extractable data (e.g., it's just static text), return:
{{"template": null, "fields_used": [], "confidence": 0.0}}
"""
        
        try:
            result = llm_client.complete_json(prompt)
            
            if result and result.get("template"):
                template = result["template"]
                confidence = result.get("confidence", 0.5)
                fields = result.get("fields_used", [])
                
                self.logger.info(f"LLM detected template with {len(fields)} fields (conf: {confidence:.2f})")
                self.logger.debug(f"Template: {template[:100]}...")
                
                return template
            
            return None
        
        except Exception as e:
            self.logger.error(f"LLM template detection failed: {e}")
            return self._detect_with_heuristics(typed_text, extraction_hints)
    
    def _detect_with_heuristics(
        self,
        typed_text: str,
        extraction_hints: List[str]
    ) -> Optional[str]:
        """Fallback heuristic-based template detection."""
        # Look for common patterns like "Label: Value"
        template = typed_text
        made_changes = False
        
        lines = typed_text.split('\n')
        new_lines = []
        
        for line in lines:
            # Pattern: "Label: Value"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Check if label matches any extraction hint
                    field_name = self._match_to_field(label, extraction_hints)
                    if field_name and value:
                        new_lines.append(f"{label}: {{{{{field_name}}}}}")
                        made_changes = True
                        continue
            
            new_lines.append(line)
        
        if made_changes:
            return '\n'.join(new_lines)
        
        return None
    
    def _match_to_field(self, label: str, extraction_hints: List[str]) -> Optional[str]:
        """Match a label to an extraction field."""
        label_lower = label.lower().replace(' ', '_')
        
        for hint in extraction_hints:
            hint_lower = hint.lower().replace(' ', '_')
            if label_lower in hint_lower or hint_lower in label_lower:
                return hint_lower
        
        # Common mappings
        mappings = {
            "name": "name",
            "restaurant": "restaurant_name",
            "rating": "rating",
            "stars": "rating",
            "address": "address",
            "location": "address",
            "phone": "phone",
            "price": "price",
            "reviews": "num_reviews",
        }
        
        for key, field in mappings.items():
            if key in label_lower:
                return field
        
        return None
    
    def detect_templates_for_steps(
        self,
        steps: List[SemanticStep],
        voice_context: Optional[VoiceContext],
        extraction_hints: List[str]
    ) -> Dict[str, str]:
        """
        Detect templates for all write steps.
        
        Returns:
            Dict mapping step_id to template string
        """
        templates = {}
        
        write_steps = [s for s in steps if s.intent == "write"]
        
        for step in write_steps:
            if not step.typed_values:
                continue
            
            typed_text = "\n".join(step.typed_values)
            
            # Skip very short text
            if len(typed_text) < 10:
                continue
            
            template = self.detect_template_llm(
                typed_text=typed_text,
                voice_context=voice_context,
                extraction_hints=extraction_hints
            )
            
            if template:
                templates[step.step_id] = template
        
        return templates