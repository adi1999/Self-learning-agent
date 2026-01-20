"""Goal-oriented workflow models for robust, intent-based automation.

Key insight: Instead of replaying ACTIONS, we achieve GOALS.
A goal has success criteria and multiple strategies to achieve it.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import re
import copy


class GoalType(str, Enum):
    """High-level goal categories"""
    NAVIGATE = "navigate"        # Get to a specific page/state
    SEARCH = "search"            # Submit a search query
    EXTRACT = "extract"          # Pull data from current page
    INPUT = "input"              # Enter data into a form/field
    SELECT = "select"            # Choose from options/click something
    SAVE = "save"                # Persist data somewhere
    LAUNCH = "launch"            # Open an application
    WRITE = "write"              # Write/type content
    SHORTCUT = "shortcut"        # Execute keyboard shortcut


class SuccessCriteria(BaseModel):
    """
    How to know if a goal was achieved.
    
    Multiple criteria can be specified - ALL must be met for success.
    """
    
    # URL-based criteria
    url_contains: Optional[str] = None          # Domain or path substring
    url_pattern: Optional[str] = None           # Regex pattern
    url_changed: bool = False                   # Just needs to change
    
    # Page content criteria  
    page_type: Optional[str] = None             # "restaurant_detail", "search_results"
    page_contains_text: Optional[str] = None    # Text that must be visible
    
    # Element-based criteria
    element_visible: Optional[str] = None       # Selector or description
    element_contains: Optional[str] = None      # Element must contain this text
    
    # Data-based criteria (for extraction)
    extracted_fields: List[str] = Field(default_factory=list)
    min_extracted_count: int = 0                # At least N fields must be extracted
    
    # App-based criteria (for desktop)
    app_active: Optional[str] = None            # App must be active
    
    # Timeout-based (always succeeds after timeout)
    timeout_success: bool = False               # Just wait and succeed
    
    def is_empty(self) -> bool:
        """Check if no criteria are set."""
        return (
            not self.url_contains and
            not self.url_pattern and
            not self.url_changed and
            not self.page_type and
            not self.page_contains_text and
            not self.element_visible and
            not self.element_contains and
            not self.extracted_fields and
            self.min_extracted_count == 0 and
            not self.app_active and
            not self.timeout_success
        )


class Strategy(BaseModel):
    """
    A single strategy to achieve a goal.
    
    Strategies are tried in priority order until success criteria are met.
    """
    
    name: str                                   # "selector_click", "gemini_vision", etc.
    priority: int = 50                          # Higher = try first (100 = highest)
    
    # Element targeting
    selector: Optional[str] = None              # CSS selector
    text_match: Optional[str] = None            # Text to find and click
    role: Optional[str] = None                  # ARIA role
    
    # Visual/Gemini targeting
    visual_description: Optional[str] = None    # Description for Gemini
    
    # Position-based fallback
    coordinates: Optional[List[int]] = None     # [x, y] fallback
    
    # For type/input strategies
    input_value: Optional[str] = None           # Value to type
    submit_after: bool = False                  # Press Enter after typing
    
    # For shortcut strategies
    shortcut_keys: Optional[str] = None         # e.g., "command+v"
    
    # Conditions for when to use this strategy
    requires_platform: Optional[str] = None     # "browser" or "desktop"
    requires_url_pattern: Optional[str] = None  # Only try on matching URLs
    
    # Accessibility (desktop)
    accessibility_role: Optional[str] = None
    accessibility_name: Optional[str] = None


class GoalStep(BaseModel):
    """
    A goal-oriented step in the workflow.
    
    Instead of "click element X", this represents "achieve goal Y".
    Success is determined by criteria, not by action completion.
    """
    
    step_id: str
    step_number: int
    
    # === THE GOAL ===
    goal_type: GoalType
    goal_description: str                       # Human-readable description
    
    # === CONTEXT ===
    platform: Literal["browser", "desktop"]
    app_name: str
    source_url_pattern: Optional[str] = None    # URL where this goal starts
    source_page_type: Optional[str] = None      # Page type where this starts
    
    # === SUCCESS CRITERIA ===
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    
    # === STRATEGIES ===
    strategies: List[Strategy] = Field(default_factory=list)
    
    # === PARAMETERS ===
    parameters: Dict[str, str] = Field(default_factory=dict)
    # e.g., {"query": "{{search_term}}", "location": "{{city}}"}
    
    # === EXTRACTION ===
    extraction_schema: Optional[Dict[str, Any]] = None
    
    # === FALLBACK BEHAVIOR ===
    fallback_to_agent: bool = True              # Let Gemini agent try if all strategies fail
    agent_goal_prompt: Optional[str] = None     # Prompt for Gemini agent
    
    # === METADATA ===
    confidence: float = 1.0
    optional: bool = False
    max_retries: int = 3
    timeout_seconds: float = 30.0
    wait_after_seconds: float = 0.5
    
    # === TEMPLATE (for write/paste goals) ===
    template: Optional[str] = None              # e.g., "{{restaurant_name}}"
    clipboard_content: Optional[str] = None     # For paste operations
    
    # === DEBUG INFO ===
    derived_from_actions: List[Dict] = Field(default_factory=list)
    original_step_id: Optional[str] = None      # Link to original WorkflowStep
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional context (e.g., navigation_intent)
    
    def get_best_strategy(self) -> Optional[Strategy]:
        """Get highest priority strategy."""
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda s: s.priority)
    
    def get_strategies_for_platform(self, platform: str) -> List[Strategy]:
        """Get strategies applicable to a platform."""
        return [
            s for s in self.strategies 
            if not s.requires_platform or s.requires_platform == platform
        ]


class GoalWorkflow(BaseModel):
    """
    Complete goal-based workflow.
    
    This replaces WorkflowRecipe for goal-oriented execution.
    """
    
    workflow_id: str
    name: str
    description: Optional[str] = None
    
    # Parameters (same as before)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Goal steps
    steps: List[GoalStep] = Field(default_factory=list)
    
    # Global extraction schema (accumulated from all extract goals)
    extraction_schema: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    version: str = "2.0"
    created_from_session: Optional[str] = None
    voice_analyzed: bool = False
    gemini_enriched: bool = False
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        return list(self.parameters.keys())
    
    def get_extraction_fields(self) -> List[str]:
        """Get all fields that will be extracted."""
        fields = set()
        for step in self.steps:
            if step.extraction_schema:
                if isinstance(step.extraction_schema, dict):
                    fields.update(step.extraction_schema.keys())
        return list(fields)
    
    def substitute_parameters(self, values: Dict[str, Any]) -> "GoalWorkflow":
        """Create a new workflow with parameters substituted."""
        workflow_copy = copy.deepcopy(self)
        
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
        
        for step in workflow_copy.steps:
            step.parameters = replace_params(step.parameters)
            if step.template:
                step.template = replace_params(step.template)
            if step.agent_goal_prompt:
                step.agent_goal_prompt = replace_params(step.agent_goal_prompt)
            if step.goal_description:
                step.goal_description = replace_params(step.goal_description)
            
            # Also substitute in strategies
            for strategy in step.strategies:
                if strategy.visual_description:
                    strategy.visual_description = replace_params(strategy.visual_description)
                if strategy.text_match:
                    strategy.text_match = replace_params(strategy.text_match)
                if strategy.input_value:
                    strategy.input_value = replace_params(strategy.input_value)
        
        return workflow_copy
    
    def fill_template(self, template: str, extracted_data: Dict[str, str]) -> str:
        """Fill a template with extracted data."""
        result = template
        for field, value in extracted_data.items():
            pattern = f"{{{{{field}}}}}"
            result = result.replace(pattern, str(value))
        return result
    
    def save(self, path):
        """Save workflow to file."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path) -> "GoalWorkflow":
        """Load workflow from file."""
        from pathlib import Path
        with open(Path(path)) as f:
            return cls.model_validate_json(f.read())

