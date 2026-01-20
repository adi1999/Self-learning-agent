"""Rule-based timeline segmentation into semantic steps."""
from typing import List, Optional, Tuple
import uuid
from src.models.session_artifact import SessionArtifact, TimelineEvent, InputEvent
from src.models.semantic_trace import SemanticStep
from src.utils.logger import setup_logger
from src.utils.config import config


class TimelineSegmenter:
    """
    Segments raw timeline into semantic steps using rule-based boundaries.
    
    Key insight: We're asking "is this the END of a user intent?"
    not "is this the START of something new?"
    
    Boundary Rules (priority order):
    1. Submit (Enter in single-line input)
    2. App switch
    3. URL/navigation change
    4. Click after idle (user made decision)
    5. Save action
    6. Typing after click (focus change)
    """
    
    def __init__(self):
        self.logger = setup_logger("Segmenter")
    
    def segment(self, session: SessionArtifact) -> List[SemanticStep]:
        """
        Segment timeline into semantic steps.
        
        Args:
            session: Raw session artifact
        
        Returns:
            List of semantic steps
        """
        if not session.timeline:
            self.logger.warning("Empty timeline, no steps to segment")
            return []
        
        steps = []
        current_segment: List[TimelineEvent] = []
        
        for i, event in enumerate(session.timeline):
            prev_event = session.timeline[i - 1] if i > 0 else None
            next_event = session.timeline[i + 1] if i + 1 < len(session.timeline) else None
            
            # Add event to current segment FIRST
            current_segment.append(event)
            
            # Then check if this event CLOSES the segment
            closes, reason = self._should_close_segment(event, prev_event, next_event, current_segment)
            
            if closes:
                step = self._create_step(current_segment, len(steps) + 1, reason)
                if step:
                    steps.append(step)
                    self.logger.debug(
                        f"Step {step.step_number}: {reason} - {step.app_name} "
                        f"({step.start_timestamp:.1f}s - {step.end_timestamp:.1f}s)"
                    )
                current_segment = []
        
        # Handle final segment
        if current_segment:
            step = self._create_step(current_segment, len(steps) + 1, "end")
            if step:
                steps.append(step)
        
        self.logger.info(f"Segmented {len(session.timeline)} events into {len(steps)} steps")
        
        return steps
    
    def _should_close_segment(
        self,
        event: TimelineEvent,
        prev_event: Optional[TimelineEvent],
        next_event: Optional[TimelineEvent],
        current_segment: List[TimelineEvent]
    ) -> Tuple[bool, str]:
        """
        Determine if current event closes the segment.
        
        Returns:
            (should_close, reason)
        """
        # Use the boundary_reason from the event if available
        if event.boundary_reason:
            if event.boundary_reason in ["submit", "app_switch", "page_load", "save"]:
                return True, event.boundary_reason
        
        # Rule 1: Check for submit action
        if self._has_submit(event):
            return True, "submit"
        
        # Rule 2: App switch
        if prev_event and event.active_app != prev_event.active_app:
            return True, "app_switch"
        
        # Rule 3: URL change (navigation)
        if prev_event and self._url_changed(prev_event, event):
            return True, "navigation"
        
        # Rule 4: Click after significant idle
        if prev_event and self._has_click(event):
            time_gap = event.timestamp - prev_event.timestamp
            if time_gap > config.click_idle_threshold:
                return True, "select"
        
        # Rule 5: Save action
        if self._has_save(event):
            return True, "save"
        
        # Rule 6: Typing after click (focus change)
        if prev_event and self._has_typing(event) and self._has_click(prev_event):
            return True, "focus_change"
        
        return False, ""
    
    def _has_submit(self, event: TimelineEvent) -> bool:
        """Check if event contains a submit action."""
        for inp in event.input_events:
            if inp.type == "keyboard" and inp.key == "return":
                # Check flush reason for context
                if inp.flush_reason == "submit":
                    return True
                # Also check if there's associated typed text
                for other in event.input_events:
                    if other.type == "keyboard" and other.text:
                        return True
        return False
    
    def _has_click(self, event: TimelineEvent) -> bool:
        """Check if event contains a click."""
        return any(inp.type == "mouse_click" for inp in event.input_events)
    
    def _has_typing(self, event: TimelineEvent) -> bool:
        """Check if event contains typed text."""
        return any(
            inp.type == "keyboard" and inp.text
            for inp in event.input_events
        )
    
    def _has_save(self, event: TimelineEvent) -> bool:
        """Check if event contains a save action."""
        return any(
            inp.type == "keyboard_shortcut" and inp.shortcut == "save"
            for inp in event.input_events
        )
    
    def _url_changed(self, prev: TimelineEvent, curr: TimelineEvent) -> bool:
        """Check if URL changed between events."""
        if not prev.url or not curr.url:
            return False
        return prev.url != curr.url
    
    def _create_step(
        self,
        events: List[TimelineEvent],
        step_number: int,
        boundary_reason: str
    ) -> Optional[SemanticStep]:
        """Create a semantic step from timeline events."""
        if not events:
            return None
        
        first = events[0]
        last = events[-1]
        
        # Collect all typed values
        typed_values = []
        for event in events:
            for inp in event.input_events:
                if inp.type == "keyboard" and inp.text:
                    typed_values.append(inp.text)
        
        # Collect clicked elements
        clicked_elements = []
        for event in events:
            for inp in event.input_events:
                if inp.type == "mouse_click":
                    click_info = {
                        "x": inp.x,
                        "y": inp.y,
                        "timestamp": inp.timestamp
                    }
                    if inp.element_info:
                        click_info["element"] = inp.element_info.model_dump()
                    clicked_elements.append(click_info)
        
        # Collect keyboard shortcuts
        shortcuts = []
        for event in events:
            for inp in event.input_events:
                if inp.type == "keyboard_shortcut" and inp.shortcut:
                    shortcuts.append(inp.shortcut)
        
        # Collect raw actions
        actions = []
        for event in events:
            for inp in event.input_events:
                action = {"type": inp.type, "timestamp": inp.timestamp}
                if inp.type == "mouse_click":
                    action["coordinates"] = (inp.x, inp.y)
                elif inp.type == "keyboard" and inp.text:
                    action["text"] = inp.text
                elif inp.type == "keyboard" and inp.key:
                    action["key"] = inp.key
                elif inp.type == "keyboard_shortcut":
                    action["shortcut"] = inp.shortcut
                actions.append(action)
        
        # Get URLs
        url_before = first.url
        url_after = last.url
        
        # Collect screenshot paths
        screenshot_paths = [
            e.screenshot_path for e in events 
            if e.screenshot_path
        ]
        
        # Generate description
        description = self._generate_description(
            first.active_app, typed_values, clicked_elements, shortcuts
        )
        
        return SemanticStep(
            step_id=f"step_{uuid.uuid4().hex[:8]}",
            step_number=step_number,
            start_timestamp=first.timestamp,
            end_timestamp=last.timestamp,
            start_event_id=first.event_id,
            end_event_id=last.event_id,
            intent="unknown",  # Will be classified later
            confidence=0.5,
            boundary_reason=boundary_reason,
            app_name=first.active_app,
            window_title=first.window_title,
            platform=first.platform,
            url_before=url_before,
            url_after=url_after,
            typed_values=typed_values,
            clicked_elements=clicked_elements,
            keyboard_shortcuts=shortcuts,
            actions=actions,
            screenshot_paths=screenshot_paths
        )
    
    def _generate_description(
        self,
        app: str,
        typed: List[str],
        clicks: List[dict],
        shortcuts: List[str]
    ) -> str:
        """Generate human-readable step description."""
        parts = []
        
        if typed:
            text_preview = " ".join(typed)[:40]
            parts.append(f"typed '{text_preview}'")
        
        if clicks:
            if len(clicks) == 1:
                parts.append("clicked element")
            else:
                parts.append(f"clicked {len(clicks)} elements")
        
        if shortcuts:
            parts.append(f"used {', '.join(shortcuts)}")
        
        if not parts:
            parts.append("action")
        
        return f"In {app}: {', '.join(parts)}"