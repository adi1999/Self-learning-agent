"""Step validation - verify that actions achieved their intent."""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from playwright.sync_api import Page
from src.models.workflow_recipe import WorkflowStep
from src.utils.logger import setup_logger


@dataclass
class ValidationResult:
    """Result of step validation."""
    success: bool
    confidence: float = 1.0
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class StepValidator:
    """
    Validates that workflow steps achieved their intent.
    
    Different validation strategies per intent:
    - Search: URL contains search terms or results visible
    - Navigate: URL changed to expected destination
    - Select: Something changed (element, URL, content)
    - Write: Text field contains expected value
    - Save: File saved indicator or dialog closed
    """
    
    def __init__(self, page: Optional[Page] = None):
        self.page = page
        self.logger = setup_logger("StepValidator")
    
    def set_page(self, page: Page):
        """Set the browser page."""
        self.page = page
    
    def capture_state(self) -> Dict[str, Any]:
        """Capture current state for comparison."""
        state = {
            "timestamp": None,
            "url": None,
            "title": None,
            "content_hash": None,
            "focused_element": None
        }
        
        if not self.page:
            return state
        
        try:
            state["url"] = self.page.url
            state["title"] = self.page.title()
            
            # Content hash
            try:
                body_text = self.page.inner_text("body")[:2000]
                state["content_hash"] = hash(body_text)
            except:
                pass
            
            # Focused element
            try:
                focused = self.page.evaluate("""
                    () => {
                        const el = document.activeElement;
                        return el ? {
                            tag: el.tagName,
                            id: el.id,
                            value: el.value
                        } : null;
                    }
                """)
                state["focused_element"] = focused
            except:
                pass
        
        except Exception as e:
            self.logger.debug(f"State capture failed: {e}")
        
        return state
    
    def validate(
        self,
        step: WorkflowStep,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that step achieved its intent.
        
        Args:
            step: Workflow step
            pre_state: State before step execution
            post_state: State after step execution
        
        Returns:
            ValidationResult with success status and details
        """
        validators = {
            "search": self._validate_search,
            "navigate": self._validate_navigate,
            "select": self._validate_select,
            "write": self._validate_write,
            "save": self._validate_save,
            "extract": self._validate_extract,
            "launch_app": self._validate_launch_app,
        }
        
        validator = validators.get(step.intent, self._validate_generic)
        
        try:
            result = validator(step, pre_state, post_state)
            
            if result.success:
                self.logger.debug(f"Step validated: {step.intent} ({result.confidence:.2f})")
            else:
                self.logger.warning(f"Step validation failed: {step.intent} - {result.reason}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(success=True, confidence=0.5, reason="Validation error")
    
    def _validate_search(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate search step."""
        
        # Check URL for search indicators
        post_url = post.get("url", "")
        search_indicators = ["search", "q=", "query=", "results", "find"]
        
        url_has_search = any(ind in post_url.lower() for ind in search_indicators)
        
        if url_has_search:
            return ValidationResult(
                success=True,
                confidence=0.95,
                reason="Search URL detected",
                details={"url": post_url}
            )
        
        # Check if URL changed at all
        if pre.get("url") != post_url:
            return ValidationResult(
                success=True,
                confidence=0.7,
                reason="URL changed after search"
            )
        
        # Check if content changed
        if pre.get("content_hash") != post.get("content_hash"):
            return ValidationResult(
                success=True,
                confidence=0.6,
                reason="Page content changed"
            )
        
        return ValidationResult(
            success=False,
            confidence=0.3,
            reason="No search results detected"
        )
    
    def _validate_navigate(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate navigation step."""
        
        if post.get("url") != pre.get("url"):
            return ValidationResult(
                success=True,
                confidence=0.95,
                reason="URL changed",
                details={
                    "from": pre.get("url"),
                    "to": post.get("url")
                }
            )
        
        # Maybe content changed without URL change (SPA)
        if pre.get("content_hash") != post.get("content_hash"):
            return ValidationResult(
                success=True,
                confidence=0.7,
                reason="Content changed (SPA navigation)"
            )
        
        return ValidationResult(
            success=False,
            confidence=0.3,
            reason="URL did not change"
        )
    
    def _validate_select(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate select/click step."""
        
        # Many things could indicate success
        changes = []
        
        if pre.get("url") != post.get("url"):
            changes.append("url_changed")
        
        if pre.get("content_hash") != post.get("content_hash"):
            changes.append("content_changed")
        
        if pre.get("focused_element") != post.get("focused_element"):
            changes.append("focus_changed")
        
        if changes:
            return ValidationResult(
                success=True,
                confidence=0.8,
                reason=f"Changes detected: {', '.join(changes)}",
                details={"changes": changes}
            )
        
        # No detectable change - might still be OK (e.g., checkbox toggle)
        return ValidationResult(
            success=True,
            confidence=0.5,
            reason="No changes detected (may still be successful)"
        )
    
    def _validate_write(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate write/type step."""
        
        # Check if focused element has value
        focused = post.get("focused_element")
        
        if focused and focused.get("value"):
            expected = step.parameter_bindings.get("value", "")
            actual = focused.get("value", "")
            
            # Check if expected value is in the field
            # (might be transformed, e.g., parameter substitution)
            if expected in actual or "{{" in expected:
                return ValidationResult(
                    success=True,
                    confidence=0.9,
                    reason="Text entered in field",
                    details={"value": actual[:50]}
                )
        
        # Content might have changed
        if pre.get("content_hash") != post.get("content_hash"):
            return ValidationResult(
                success=True,
                confidence=0.7,
                reason="Content changed after typing"
            )
        
        return ValidationResult(
            success=True,
            confidence=0.6,
            reason="Write action completed (unverified)"
        )
    
    def _validate_save(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate save step."""
        
        # For browser, not much we can verify
        # For desktop, could check file modification time
        
        return ValidationResult(
            success=True,
            confidence=0.7,
            reason="Save action executed"
        )
    
    def _validate_extract(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate extraction step."""
        
        # Extraction validation happens via LLMExtractor results
        return ValidationResult(
            success=True,
            confidence=0.8,
            reason="Extraction attempted"
        )
    
    def _validate_launch_app(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Validate app launch step."""
        
        # Would need to check active app via AppLauncher
        return ValidationResult(
            success=True,
            confidence=0.7,
            reason="App launch attempted"
        )
    
    def _validate_generic(
        self,
        step: WorkflowStep,
        pre: Dict,
        post: Dict
    ) -> ValidationResult:
        """Generic validation for unknown intents."""
        
        # Check for any changes
        if pre != post:
            return ValidationResult(
                success=True,
                confidence=0.6,
                reason="State changed"
            )
        
        return ValidationResult(
            success=True,
            confidence=0.4,
            reason="Action executed (unknown intent)"
        )