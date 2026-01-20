"""Intent classification for semantic steps."""
from typing import Dict, Any, List
from src.models.semantic_trace import SemanticStep
from src.utils.logger import setup_logger
from src.utils.llm_client import llm_client


class IntentClassifier:
    """
    Classifies step intent using heuristics and LLM fallback.
    
    Intent types:
    - search: Searching for information
    - select: Clicking to choose something
    - navigate: Moving to a new page/location
    - write: Entering text content
    - extract: Copying/reading information
    - save: Saving work
    - launch_app: Switching applications
    """
    
    KNOWN_INTENTS = [
        "search", "select", "navigate", "write",
        "extract", "save", "launch_app", "unknown"
    ]
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize intent classifier.
        
        Args:
            use_llm: Use LLM for ambiguous cases
        """
        self.use_llm = use_llm and llm_client.is_available
        self.logger = setup_logger("IntentClassifier")
    
    def classify(self, step: SemanticStep) -> Dict[str, Any]:
        """
        Classify step intent.
        
        Args:
            step: Semantic step to classify
        
        Returns:
            Dict with intent, confidence, and reasoning
        """
        # Try heuristics first
        result = self._classify_with_heuristics(step)
        
        # Use LLM for low-confidence or unknown
        if self.use_llm and (result["confidence"] < 0.6 or result["intent"] == "unknown"):
            llm_result = self._classify_with_llm(step)
            if llm_result and llm_result.get("confidence", 0) > result["confidence"]:
                result = llm_result
        
        return result
    
    def _classify_with_heuristics(self, step: SemanticStep) -> Dict[str, Any]:
        """Classify using rule-based heuristics."""
        
        # Collect signals
        has_typing = step.has_typing()
        has_click = step.has_clicks()
        has_shortcuts = step.has_shortcuts()
        is_browser = step.platform == "browser"
        url_changed = step.url_before != step.url_after and step.url_after
        boundary = step.boundary_reason
        
        # Check for specific shortcuts
        has_copy = "copy" in step.keyboard_shortcuts
        has_paste = "paste" in step.keyboard_shortcuts
        has_save = "save" in step.keyboard_shortcuts
        
        # Pattern: App switch
        if boundary == "app_switch":
            return {
                "intent": "launch_app",
                "confidence": 0.9,
                "reasoning": "Application switched"
            }
        
        # Pattern: Save shortcut
        if has_save or boundary == "save":
            return {
                "intent": "save",
                "confidence": 0.95,
                "reasoning": "Save action detected"
            }
        
        # Pattern: Copy action → Extract
        if has_copy:
            return {
                "intent": "extract",
                "confidence": 0.85,
                "reasoning": "Copy action indicates extraction"
            }
        
        # Pattern: Type + Submit in browser → Search
        if has_typing and boundary == "submit" and is_browser:
            # Check if URL looks like search
            url = step.url_after or step.url_before or ""
            is_search_url = any(x in url.lower() for x in ["search", "q=", "query=", "google", "bing"])
            
            if is_search_url:
                return {
                    "intent": "search",
                    "confidence": 0.95,
                    "reasoning": "Typed and submitted in browser, search URL detected"
                }
            return {
                "intent": "search",
                "confidence": 0.85,
                "reasoning": "Typed and submitted in browser"
            }
        
        # Pattern: URL changed without typing → Navigate (clicked a link)
        if url_changed and not has_typing:
            return {
                "intent": "navigate",
                "confidence": 0.9,
                "reasoning": "URL changed from click action"
            }
        
        # Pattern: Click only, no typing, no URL change → Select
        if has_click and not has_typing and not url_changed:
            return {
                "intent": "select",
                "confidence": 0.8,
                "reasoning": "Clicked element without typing"
            }
        
        # Pattern: Typing in desktop app → Write
        if has_typing and not is_browser:
            return {
                "intent": "write",
                "confidence": 0.85,
                "reasoning": "Typing in desktop application"
            }
        
        # Pattern: Typing in browser but not submitted
        if has_typing and is_browser and boundary != "submit":
            return {
                "intent": "write",
                "confidence": 0.7,
                "reasoning": "Typing in browser without submit"
            }
        
        # Pattern: Paste action → Write (pasting content)
        if has_paste:
            return {
                "intent": "write",
                "confidence": 0.8,
                "reasoning": "Paste action detected"
            }
        
        # Fallback
        return {
            "intent": "unknown",
            "confidence": 0.3,
            "reasoning": "Could not determine intent from signals"
        }
    
    def _classify_with_llm(self, step: SemanticStep) -> Dict[str, Any]:
        """Classify using LLM for ambiguous cases."""
        
        prompt = f"""Classify this user action into a workflow step intent.

## Context
- Application: {step.app_name}
- Platform: {step.platform}
- Window Title: {step.window_title}
- URL (before): {step.url_before or "N/A"}
- URL (after): {step.url_after or "N/A"}
- Boundary Reason: {step.boundary_reason}
- Duration: {step.duration:.1f}s

## Actions Performed
- Typed text: {step.typed_values if step.typed_values else "None"}
- Clicks: {len(step.clicked_elements)} element(s)
- Keyboard shortcuts: {step.keyboard_shortcuts if step.keyboard_shortcuts else "None"}
- Submit/Enter: {"Yes" if step.boundary_reason == "submit" else "No"}

## Voice Context
{step.voice_transcript or "No voice narration"}

## Available Intents
- search: User searching for information (typed query + submitted)
- select: User clicking to choose something (link, button, option)
- navigate: User moving to a new page/location
- write: User entering text content (notes, forms, documents)
- extract: User copying/reading information to use elsewhere
- save: User saving their work
- launch_app: User switching to a different application

## Examples
1. Typed "sushi restaurants" + Enter in Chrome/Google → search
2. Clicked on a search result link → select (or navigate if URL changed)
3. Typed restaurant details in Notes → write
4. Pressed Cmd+S → save
5. Used Cmd+C to copy text → extract

Respond with JSON only:
{{"intent": "...", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
        
        try:
            result = llm_client.complete_json(prompt)
            
            if result and result.get("intent") in self.KNOWN_INTENTS:
                self.logger.debug(f"LLM classified as: {result['intent']} ({result.get('confidence', 0):.2f})")
                return result
        
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
        
        return None
    
    def classify_batch(self, steps: List[SemanticStep]) -> List[Dict[str, Any]]:
        """Classify multiple steps."""
        results = []
        for step in steps:
            result = self.classify(step)
            results.append(result)
        return results