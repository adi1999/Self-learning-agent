"""Completion detection for workflow steps."""
import time
from typing import Optional
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout
from src.models.workflow_recipe import CompletionSignal, WorkflowStep
from src.utils.logger import setup_logger


class CompletionDetector:
    """
    Detects when a workflow step has completed.
    
    Different strategies for different step types:
    - Search: Wait for URL change or results to appear
    - Navigate: Wait for page load
    - Select: Wait for content change
    - Write: Wait for element stable
    """
    
    def __init__(self, page: Optional[Page] = None):
        self.page = page
        self.logger = setup_logger("CompletionDetector")
    
    def set_page(self, page: Page):
        """Set the browser page."""
        self.page = page
    
    def wait_for_completion(
        self,
        step: WorkflowStep,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait for step completion based on completion signal.
        
        Args:
            step: Workflow step
            timeout: Override timeout (seconds)
        
        Returns:
            True if completed successfully
        """
        signal = step.completion_signal
        
        if not signal:
            # Default based on intent
            return self._wait_by_intent(step, timeout)
        
        timeout_ms = timeout * 1000 if timeout else signal.timeout_ms
        
        if signal.type == "url_change":
            return self._wait_for_url_change(timeout_ms)
        
        elif signal.type == "element_visible":
            return self._wait_for_element_visible(signal.selector, timeout_ms)
        
        elif signal.type == "element_hidden":
            return self._wait_for_element_hidden(signal.selector, timeout_ms)
        
        elif signal.type == "network_idle":
            return self._wait_for_network_idle(timeout_ms)
        
        elif signal.type == "content_change":
            return self._wait_for_content_change(timeout_ms)
        
        elif signal.type == "timeout":
            time.sleep(timeout_ms / 1000)
            return True
        
        return True
    
    def _wait_by_intent(self, step: WorkflowStep, timeout: Optional[float]) -> bool:
        """Wait based on step intent."""
        timeout_ms = (timeout or 10.0) * 1000
        
        if step.intent == "search":
            return self._wait_for_search_results(timeout_ms)
        
        elif step.intent == "navigate":
            return self._wait_for_page_load(timeout_ms)
        
        elif step.intent == "select":
            return self._wait_for_content_change(timeout_ms)
        
        elif step.intent in ["write", "save"]:
            time.sleep(0.5)
            return True
        
        # Default short wait
        time.sleep(0.3)
        return True
    
    def _wait_for_url_change(self, timeout_ms: int) -> bool:
        """Wait for URL to change."""
        if not self.page:
            return True
        
        try:
            initial_url = self.page.url
            
            start = time.time()
            while (time.time() - start) * 1000 < timeout_ms:
                if self.page.url != initial_url:
                    # Also wait for load
                    self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                    return True
                time.sleep(0.1)
            
            return False
        except Exception as e:
            self.logger.debug(f"URL change wait failed: {e}")
            return True
    
    def _wait_for_element_visible(self, selector: str, timeout_ms: int) -> bool:
        """Wait for element to become visible."""
        if not self.page or not selector:
            return True
        
        try:
            self.page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
            return True
        except PlaywrightTimeout:
            return False
    
    def _wait_for_element_hidden(self, selector: str, timeout_ms: int) -> bool:
        """Wait for element to become hidden."""
        if not self.page or not selector:
            return True
        
        try:
            self.page.wait_for_selector(selector, state="hidden", timeout=timeout_ms)
            return True
        except PlaywrightTimeout:
            return False
    
    def _wait_for_network_idle(self, timeout_ms: int) -> bool:
        """Wait for network to be idle."""
        if not self.page:
            return True
        
        try:
            self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
            return True
        except PlaywrightTimeout:
            # Try domcontentloaded as fallback
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=timeout_ms // 2)
                return True
            except:
                return False
    
    def _wait_for_page_load(self, timeout_ms: int) -> bool:
        """Wait for page to finish loading."""
        if not self.page:
            return True
        
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            return True
        except PlaywrightTimeout:
            return False
    
    def _wait_for_content_change(self, timeout_ms: int) -> bool:
        """Wait for page content to change."""
        if not self.page:
            time.sleep(timeout_ms / 1000)
            return True
        
        try:
            # Get initial content hash
            initial_hash = self._get_content_hash()
            
            start = time.time()
            while (time.time() - start) * 1000 < timeout_ms:
                current_hash = self._get_content_hash()
                if current_hash != initial_hash:
                    return True
                time.sleep(0.2)
            
            # No change detected, but not necessarily a failure
            return True
        except Exception:
            return True
    
    def _wait_for_search_results(self, timeout_ms: int) -> bool:
        """Wait for search results to appear."""
        if not self.page:
            return True
        
        # Common search result selectors
        result_selectors = [
            '#search',           # Google
            '.g',                # Google results
            '[data-result]',
            '.search-result',
            '.results',
            '#results',
            'main article',
            '.organic-result',
        ]
        
        for selector in result_selectors:
            try:
                self.page.wait_for_selector(selector, state="visible", timeout=timeout_ms // len(result_selectors))
                return True
            except PlaywrightTimeout:
                continue
        
        # Fallback to network idle
        return self._wait_for_network_idle(timeout_ms // 2)
    
    def _get_content_hash(self) -> str:
        """Get a hash of page content for change detection."""
        if not self.page:
            return ""
        
        try:
            text = self.page.inner_text("body")
            return str(hash(text[:1000]))
        except:
            return ""