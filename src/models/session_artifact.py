"""Session Artifact - Raw recording output from Observer."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pathlib import Path
import json
import uuid


class ElementInfo(BaseModel):
    """Information about a UI element that was interacted with."""
    
    # Browser fields
    tag: Optional[str] = None
    element_id: Optional[str] = None
    classes: Optional[List[str]] = None
    name: Optional[str] = None
    input_type: Optional[str] = None
    role: Optional[str] = None
    aria_label: Optional[str] = None
    placeholder: Optional[str] = None
    text: Optional[str] = None
    href: Optional[str] = None
    selector: Optional[str] = None
    is_content_editable: Optional[bool] = None
    
    # Desktop fields (macapptree)
    accessibility_role: Optional[str] = None
    accessibility_name: Optional[str] = None
    accessibility_description: Optional[str] = None
    bbox: Optional[List[int]] = None  # [x, y, width, height]
    absolute_position: Optional[str] = None
    tree_element_id: Optional[str] = None
    
    def get_best_identifier(self) -> str:
        """Get the best available identifier for this element."""
        if self.selector:
            return f"selector:{self.selector}"
        if self.element_id:
            return f"id:{self.element_id}"
        if self.accessibility_role and self.accessibility_name:
            return f"a11y:{self.accessibility_role}[{self.accessibility_name}]"
        if self.text:
            return f"text:{self.text[:30]}"
        if self.role:
            return f"role:{self.role}"
        return "unknown"


class InputEvent(BaseModel):
    """A single input event (keyboard or mouse)."""
    
    type: Literal["mouse_click", "mouse_move", "keyboard", "keyboard_shortcut", "scroll"]
    timestamp: float  # Seconds since recording start
    
    # Mouse fields
    x: Optional[int] = None
    y: Optional[int] = None
    button: Optional[str] = None  # left, right, middle
    
    # Keyboard fields
    text: Optional[str] = None  # Typed text (buffered)
    key: Optional[str] = None  # Special key (return, tab, escape)
    shortcut: Optional[str] = None  # copy, paste, save, etc.
    
    # Context
    element_info: Optional[ElementInfo] = None
    flush_reason: Optional[str] = None  # Why typing was flushed

    # Clipboard content (captured on copy)
    clipboard_content: Optional[str] = None


class NavigationOutcome(BaseModel):
    """Tracks what happened AFTER an action (e.g., URL change after click)."""
    
    url_before: Optional[str] = None       # URL when action started
    url_after: Optional[str] = None        # URL after action completed
    domain_before: Optional[str] = None    # Domain before
    domain_after: Optional[str] = None     # Domain after
    domain_changed: bool = False           # Did the domain change?
    page_title_after: Optional[str] = None # Page title after navigation
    navigation_type: Optional[str] = None  # "same_page", "same_domain", "cross_domain"
    wait_time_ms: int = 0                  # How long we waited for navigation


class TimelineEvent(BaseModel):
    """A point in the recorded timeline."""
    
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: float  # Seconds since recording start
    
    # Context
    active_app: str
    window_title: str
    platform: Literal["browser", "desktop"]
    url: Optional[str] = None  # Browser only
    
    # Captures
    screenshot_path: Optional[str] = None
    dom_snapshot_path: Optional[str] = None
    
    # Actions in this event
    input_events: List[InputEvent] = Field(default_factory=list)
    
    # Boundary info
    boundary_reason: Optional[str] = None  # click, submit, app_switch, copy, etc.
    
    # Voice segment (if any voice during this event) - set during compile
    voice_segment: Optional[str] = None
    
    # === NEW: Navigation outcome tracking ===
    navigation_outcome: Optional[NavigationOutcome] = None


class VoiceTranscription(BaseModel):
    """Voice transcription with segments (populated during compile phase)."""
    
    text: str
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None


class SessionArtifact(BaseModel):
    """Complete recording session - raw capture from Observer."""
    
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    platform: Literal["macos", "windows", "linux"] = "macos"
    
    # Timing
    recording_start: datetime = Field(default_factory=datetime.utcnow)
    recording_end: Optional[datetime] = None
    
    # Timeline
    timeline: List[TimelineEvent] = Field(default_factory=list)
    
    # Voice - raw audio path (transcription done at compile time)
    voice_audio_path: Optional[str] = None
    
    # Voice transcription (populated during compile phase, not recording)
    voice_transcription: Optional[VoiceTranscription] = None
    
    # Metadata
    browser_used: Optional[str] = None  # chromium, firefox, etc.
    initial_url: Optional[str] = None
    
    def add_event(self, event: TimelineEvent):
        """Add a timeline event."""
        self.timeline.append(event)
    
    def finalize(self):
        """Mark recording as complete."""
        self.recording_end = datetime.utcnow()
    
    def duration(self) -> float:
        """Get recording duration in seconds."""
        if not self.timeline:
            return 0.0
        return self.timeline[-1].timestamp
    
    def get_events_in_range(self, start: float, end: float) -> List[TimelineEvent]:
        """Get events within a time range."""
        return [e for e in self.timeline if start <= e.timestamp <= end]
    
    def get_all_typed_text(self) -> List[str]:
        """Get all typed text from the session."""
        texts = []
        for event in self.timeline:
            for inp in event.input_events:
                if inp.type == "keyboard" and inp.text:
                    texts.append(inp.text)
        return texts
    
    def get_copy_events(self) -> List[Dict[str, Any]]:
        """Get all copy events with their clipboard content and context."""
        copies = []
        for event in self.timeline:
            for inp in event.input_events:
                if inp.shortcut == "copy" and inp.clipboard_content:
                    copies.append({
                        "timestamp": inp.timestamp,
                        "clipboard_content": inp.clipboard_content,
                        "app": event.active_app,
                        "url": event.url,
                        "screenshot_path": event.screenshot_path
                    })
        return copies
    
    def save(self, directory: Path) -> Path:
        """
        Save session artifact to disk.
        
        Structure:
            {directory}/{session_id}/
                metadata.json
                screenshots/
                voice/
        """
        session_dir = directory / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (session_dir / "screenshots").mkdir(exist_ok=True)
        (session_dir / "voice").mkdir(exist_ok=True)
        
        # Save metadata
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
        
        return session_dir
    
    @classmethod
    def load(cls, directory: Path) -> "SessionArtifact":
        """Load session artifact from disk."""
        metadata_path = directory / "metadata.json"
        with open(metadata_path) as f:
            return cls.model_validate_json(f.read())