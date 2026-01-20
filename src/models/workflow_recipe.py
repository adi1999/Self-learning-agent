"""Workflow Recipe - Reusable, parameterized workflow definition."""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from pathlib import Path
import re
import copy


class WorkflowParameter(BaseModel):
    """A parameter that can vary between workflow executions."""
    
    name: str
    param_type: Literal["string", "number", "boolean"] = "string"
    description: Optional[str] = None
    example_value: Any
    required: bool = True
    default_value: Optional[Any] = None


class ExtractionField(BaseModel):
    """Schema for a single field to extract."""
    
    description: str
    visual_hint: Optional[str] = None  # From Gemini: "large heading at top"
    example_value: Optional[str] = None  # Value seen during recording
    required: bool = True


class ExtractionSchema(BaseModel):
    """Complete extraction schema enriched by Gemini."""
    
    fields: Dict[str, ExtractionField] = Field(default_factory=dict)
    page_type: Optional[str] = None  # "restaurant_detail", "search_results"
    page_source: Optional[str] = None  # "yelp", "google"
    layout_hints: Optional[str] = None  # General page layout description
    
    def to_simple_dict(self) -> Dict[str, str]:
        """Convert to simple dict for backward compatibility."""
        return {name: field.description for name, field in self.fields.items()}
    
    def to_gemini_schema(self) -> Dict[str, Any]:
        """Convert to format expected by Gemini extraction."""
        return {
            name: {
                "description": field.description,
                "visual_hint": field.visual_hint
            }
            for name, field in self.fields.items()
        }


class ElementReference(BaseModel):
    """Multi-strategy element reference for robust element location."""
    
    # Browser strategies
    selector: Optional[str] = None
    
    # Accessibility strategies (both browser and desktop)
    role: Optional[str] = None
    name: Optional[str] = None
    aria_label: Optional[str] = None
    
    # Desktop strategies (macapptree)
    accessibility_role: Optional[str] = None
    accessibility_name: Optional[str] = None
    
    # Text-based
    text: Optional[str] = None
    placeholder: Optional[str] = None
    
    # Visual/positional (from recording)
    bbox: Optional[List[int]] = None
    absolute_position: Optional[str] = None
    coordinates: Optional[List[int]] = None  # [x, y]
    
    # Visual reference from Gemini (normalized 0-999)
    visual_region: Optional[List[int]] = None  # [x1, y1, x2, y2] normalized
    visual_hint: Optional[str] = None  # "blue button in top right"
    
    # For search inputs specifically
    is_search_input: bool = False
    
    def has_strategies(self) -> bool:
        """Check if at least one strategy is available."""
        return any([
            self.selector,
            self.role,
            self.accessibility_role,
            self.text,
            self.coordinates,
            self.visual_region
        ])
    
    def get_description(self) -> str:
        """Get human-readable description."""
        parts = []
        if self.role:
            parts.append(f"role={self.role}")
        if self.text:
            parts.append(f'text="{self.text[:20]}"')
        if self.selector:
            parts.append(f"selector={self.selector[:30]}")
        if self.accessibility_role:
            parts.append(f"a11y={self.accessibility_role}")
        if self.visual_hint:
            parts.append(f"visual={self.visual_hint[:30]}")
        return " | ".join(parts) if parts else "unknown element"


class CompletionSignal(BaseModel):
    """How to detect step completion."""
    
    type: Literal["url_change", "element_visible", "element_hidden", "network_idle", "timeout", "content_change"]
    selector: Optional[str] = None
    url_pattern: Optional[str] = None
    required_page_type: Optional[str] = None  # NEW: Enforce landing on specific page type (e.g., 'restaurant_detail')
    timeout_ms: int = 10000


class WorkflowStep(BaseModel):
    """A single step in the workflow."""
    
    step_id: str
    step_number: int
    intent: str  # search, select, navigate, write, extract, save, launch_app
    description: str
    
    # Context
    platform: Literal["browser", "desktop"]
    app_name: str
    
    # Action
    action_type: Literal["click", "type", "navigate", "extract", "save", "launch_app", "shortcut", "wait"]
    
    # Element targeting (for click/type actions)
    element_reference: Optional[ElementReference] = None
    
    # Parameter bindings
    parameter_bindings: Dict[str, str] = Field(default_factory=dict)
    # e.g., {"value": "{{cuisine}} restaurants in {{location}}"}
    
    # For extract steps - ENHANCED with Gemini analysis
    extraction_schema: Optional[ExtractionSchema] = None
    
    # For write steps
    template: Optional[str] = None
    # e.g., "Restaurant: {{restaurant_name}}\nRating: {{rating}}"
    
    # For shortcut steps
    shortcut: Optional[str] = None  # save, copy, paste, etc.
    
    # Clipboard content to paste (captured during recording)
    clipboard_content: Optional[str] = None
    
    # Completion detection
    completion_signal: Optional[CompletionSignal] = None
    
    # Timing
    wait_before_ms: int = 0
    wait_after_ms: int = 500
    
    # Metadata
    confidence: float = 1.0
    optional: bool = False
    
    # Original screenshot for reference (used by Gemini)
    screenshot_path: Optional[str] = None
    
    # Page context from Gemini analysis
    page_context: Optional[Dict[str, Any]] = None
    
    # Navigation goal: What URL pattern should we see after a successful click?
    # e.g., "zomato.com", "yelp.com/biz/"
    expected_url_pattern: Optional[str] = None


class FailurePolicy(BaseModel):
    """How to handle failures during execution."""
    
    retry_limit: int = 2
    retry_delay_ms: int = 1000
    on_failure: Literal["abort", "skip", "retry_then_abort", "retry_then_skip"] = "retry_then_abort"
    min_step_confidence: float = 0.3
    use_gemini_fallback: bool = True  # Try Gemini if deterministic fails


class WorkflowRecipe(BaseModel):
    """Complete workflow definition that can be replayed with different parameters."""
    
    workflow_id: str
    name: str
    description: Optional[str] = None
    
    # Parameters
    parameters: Dict[str, WorkflowParameter] = Field(default_factory=dict)
    
    # Steps
    steps: List[WorkflowStep] = Field(default_factory=list)
    
    # Failure handling
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)
    
    # Metadata
    created_from_session: Optional[str] = None
    version: str = "1.0"
    
    # Compilation info
    voice_analyzed: bool = False
    gemini_enriched: bool = False
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        return [name for name, param in self.parameters.items() if param.required]
    
    def get_extraction_fields(self) -> List[str]:
        """Get all fields that will be extracted."""
        fields = set()
        for step in self.steps:
            if step.extraction_schema and step.extraction_schema.fields:
                fields.update(step.extraction_schema.fields.keys())
        return list(fields)
    
    def validate_parameters(self, values: Dict[str, Any]) -> List[str]:
        """
        Validate provided parameter values.
        
        Returns list of error messages (empty if valid).
        """
        errors = []
        
        # Check required parameters
        for name, param in self.parameters.items():
            if param.required and name not in values:
                errors.append(f"Missing required parameter: {name}")
        
        # Check types
        for name, value in values.items():
            if name not in self.parameters:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            param = self.parameters[name]
            if param.param_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Parameter '{name}' must be a number")
            elif param.param_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Parameter '{name}' must be a boolean")
        
        return errors
    
    def substitute_parameters(self, values: Dict[str, Any]) -> "WorkflowRecipe":
        """
        Create a new recipe with parameters substituted.
        
        Replaces all {{param_name}} references with actual values.
        """
        recipe_copy = copy.deepcopy(self)
        
        def replace_params(obj):
            if isinstance(obj, str):
                for param_name, param_value in values.items():
                    pattern = f"{{{{{param_name}}}}}"
                    obj = obj.replace(pattern, str(param_value))
                return obj
            elif isinstance(obj, dict):
                return {k: replace_params(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_params(item) for item in obj]
            return obj
        
        for step in recipe_copy.steps:
            step.parameter_bindings = replace_params(step.parameter_bindings)
            if step.template:
                step.template = replace_params(step.template)
            if step.element_reference:
                step.element_reference = ElementReference(**replace_params(step.element_reference.model_dump()))
        
        return recipe_copy
    
    def fill_template(self, template: str, extracted_data: Dict[str, str]) -> str:
        """
        Fill a template with extracted data.
        
        Args:
            template: Template string with {{field}} placeholders
            extracted_data: Dict of extracted values
        
        Returns:
            Filled template string
        """
        result = template
        for field, value in extracted_data.items():
            pattern = f"{{{{{field}}}}}"
            result = result.replace(pattern, str(value))
        return result
    
    def get_step_by_intent(self, intent: str) -> List[WorkflowStep]:
        """Get all steps with specific intent."""
        return [s for s in self.steps if s.intent == intent]
    
    def save(self, path: Path):
        """Save recipe to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "WorkflowRecipe":
        """Load recipe from file."""
        with open(path) as f:
            return cls.model_validate_json(f.read())