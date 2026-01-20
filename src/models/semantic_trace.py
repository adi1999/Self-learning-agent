"""Semantic Trace - Interpreted recording with semantic understanding."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path


class SemanticStep(BaseModel):
    """A semantic step extracted from raw timeline."""
    
    step_id: str
    step_number: int
    
    # Timing
    start_timestamp: float
    end_timestamp: float
    start_event_id: str
    end_event_id: str
    
    # Classification
    intent: str  # search, select, navigate, write, extract, save, launch_app
    confidence: float = 0.5
    boundary_reason: str  # submit, app_switch, navigation, select, save, focus_change
    
    # Context
    app_name: str
    window_title: str
    platform: str  # browser or desktop
    url_before: Optional[str] = None
    url_after: Optional[str] = None
    domain_after: Optional[str] = None  # Target domain after navigation
    
    # Actions
    typed_values: List[str] = Field(default_factory=list)
    clicked_elements: List[Dict[str, Any]] = Field(default_factory=list)
    keyboard_shortcuts: List[str] = Field(default_factory=list)
    
    # Raw actions for reference
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Voice context (if voice during this step)
    voice_transcript: Optional[str] = None
    
    # Extraction hints (from voice or inference)
    extraction_hints: Optional[List[str]] = None
    
    # Screenshots for this step
    screenshot_paths: List[str] = Field(default_factory=list)
    
    # LLM analysis (if used)
    llm_analysis: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """Step duration in seconds."""
        return self.end_timestamp - self.start_timestamp
    
    def has_typing(self) -> bool:
        """Check if step has typed values."""
        return len(self.typed_values) > 0
    
    def has_clicks(self) -> bool:
        """Check if step has clicked elements."""
        return len(self.clicked_elements) > 0
    
    def has_shortcuts(self) -> bool:
        """Check if step has keyboard shortcuts."""
        return len(self.keyboard_shortcuts) > 0
    
    def get_combined_typed_text(self) -> str:
        """Get all typed text combined."""
        return " ".join(self.typed_values)


class ParameterCandidate(BaseModel):
    """A value that might be a parameter."""
    
    value: str
    suggested_name: str
    param_type: str = "string"  # string, number, boolean
    confidence: float = 0.5
    
    # Evidence
    source_step_ids: List[str] = Field(default_factory=list)
    was_in_search: bool = False
    was_in_voice: bool = False
    semantic_type: Optional[str] = None  # cuisine, location, name, query, etc.
    
    # Context
    context_app: Optional[str] = None
    context_url: Optional[str] = None


class VoiceContext(BaseModel):
    """Extracted context from voice narration."""
    
    # Field labels mentioned
    field_labels: List[Dict[str, Any]] = Field(default_factory=list)
    # e.g., [{"phrase": "restaurant name", "field_name": "restaurant_name", "timestamp": 5.2}]
    
    # Parameter hints
    parameter_hints: List[Dict[str, Any]] = Field(default_factory=list)
    # e.g., [{"value": "sushi", "type": "cuisine", "timestamp": 3.0}]
    
    # Connections between steps
    task_connections: List[Dict[str, Any]] = Field(default_factory=list)
    # e.g., [{"source": "weather", "target": "jacket search", "relationship": "..."}]
    
    # Extraction hints
    extraction_hints: List[str] = Field(default_factory=list)
    # e.g., ["restaurant name", "rating", "address"]
    
    # Overall task goal
    task_goal: Optional[str] = None


class SemanticTrace(BaseModel):
    """Complete interpreted recording with semantic understanding."""
    
    session_id: str
    
    # Semantic steps
    steps: List[SemanticStep] = Field(default_factory=list)
    
    # Voice context
    voice_context: Optional[VoiceContext] = None
    
    # Parameter candidates
    parameter_candidates: List[ParameterCandidate] = Field(default_factory=list)
    
    # Metadata
    segmentation_method: str = "rule_based"  # rule_based, llm_guided
    classification_method: str = "heuristics"  # heuristics, llm
    total_duration: float = 0.0
    
    def get_steps_by_intent(self, intent: str) -> List[SemanticStep]:
        """Get all steps with a specific intent."""
        return [s for s in self.steps if s.intent == intent]
    
    def get_step_by_id(self, step_id: str) -> Optional[SemanticStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_extraction_steps(self) -> List[SemanticStep]:
        """Get steps that involve extraction."""
        return [s for s in self.steps if s.intent == "extract" or s.extraction_hints]
    
    def get_high_confidence_parameters(self, threshold: float = 0.6) -> List[ParameterCandidate]:
        """Get parameter candidates above threshold."""
        return [p for p in self.parameter_candidates if p.confidence >= threshold]
    
    def save(self, path: Path):
        """Save trace to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "SemanticTrace":
        """Load trace from file."""
        with open(path) as f:
            return cls.model_validate_json(f.read())