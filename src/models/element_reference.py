"""ElementReference - Multi-strategy element identification."""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class AccessibilityInfo(BaseModel):
    """Accessibility attributes for element identification."""
    role: Optional[str] = None
    label: Optional[str] = None
    name: Optional[str] = None


class ElementReference(BaseModel):
    """
    Multi-strategy element reference for robust element location.
    
    During recording: Capture everything we can about the element
    During replay: Try strategies in order until one works
    """
    # Strategy 1: Accessibility (highest confidence)
    accessibility: Optional[AccessibilityInfo] = None
    
    # Strategy 2: DOM selector (browser only)
    dom_selector: Optional[str] = None
    
    # Strategy 3: Text content
    text: Optional[str] = None
    
    # Strategy 4: Visual region (fallback)
    visual_region: Optional[list[int]] = None  # [x, y, width, height]
    
    # Strategy 5: Spatial hints
    spatial_hint: Optional[str] = None  # "top-left", "center", etc.
    
    # Metadata
    element_type: Optional[str] = None  # "button", "input", "link"
    context: Optional[str] = None  # Nearby text for disambiguation
    
    def has_strategies(self) -> bool:
        """Check if at least one strategy is available."""
        return any([
            self.accessibility is not None,
            self.dom_selector is not None,
            self.text is not None,
            self.visual_region is not None
        ])
    
    def get_description(self) -> str:
        """Get human-readable description of element."""
        parts = []
        
        if self.element_type:
            parts.append(self.element_type)
        
        if self.text:
            parts.append(f'"{self.text[:30]}"')
        elif self.accessibility and self.accessibility.label:
            parts.append(f'"{self.accessibility.label[:30]}"')
        
        if self.dom_selector:
            parts.append(f"selector: {self.dom_selector[:30]}")
        
        return " ".join(parts) if parts else "unknown element"


class ActionType(BaseModel):
    """Types of actions that can be performed on elements."""
    type: Literal["click", "type", "select", "hover", "scroll"]
    value: Optional[str] = None  # For type/select actions
    button: Optional[str] = None  # For click actions


class ElementAction(BaseModel):
    """An action to perform on an element."""
    element: ElementReference
    action: ActionType
    wait_after: float = 0.5  # Seconds to wait after action
    
    def describe(self) -> str:
        """Get human-readable description."""
        action_desc = self.action.type
        if self.action.value:
            action_desc += f" '{self.action.value[:30]}'"
        
        elem_desc = self.element.get_description()
        return f"{action_desc} on {elem_desc}"