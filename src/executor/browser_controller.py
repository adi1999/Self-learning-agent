"""Browser automation using Playwright."""
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from typing import Optional
from pathlib import Path
import time


class BrowserController:
    """Controls browser automation using Playwright."""
    
    def __init__(self, headless: bool = False):
        """
        Initialize browser controller.
        
        Args:
            headless: Run browser in headless mode
        """
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
    
    def launch(self, url: Optional[str] = None):
        """
        Launch browser and optionally navigate to URL.
        
        Args:
            url: URL to navigate to after launch
        """
        self.playwright = sync_playwright().start()
        
        # Launch Chromium (works with Brave/Chrome too)
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            channel="chrome"  # Use system Chrome if available
        )
        
        # Create new page
        self.page = self.browser.new_page()
        
        # Set reasonable timeouts
        self.page.set_default_timeout(10000)  # 10 seconds
        
        if url:
            self.page.goto(url)
    
    def navigate(self, url: str, wait_until: str = "domcontentloaded"):
        """
        Navigate to URL.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation successful
        """
        if not self.page:
            raise RuntimeError("Browser not launched")
        
        self.page.goto(url, wait_until=wait_until)
    
    def get_current_url(self) -> str:
        """Get current page URL."""
        if not self.page:
            return ""
        return self.page.url
    
    def get_title(self) -> str:
        """Get current page title."""
        if not self.page:
            return ""
        return self.page.title()
    
    def wait_for_navigation(self, timeout: float = 5.0):
        """Wait for page navigation to complete."""
        if not self.page:
            return
        
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
        except PlaywrightTimeout:
            pass  # Continue anyway
    
    def wait_for_selector(self, selector: str, timeout: float = 5.0) -> bool:
        """
        Wait for element matching selector to appear.
        
        Returns:
            True if element appeared, False if timeout
        """
        if not self.page:
            return False
        
        try:
            self.page.wait_for_selector(selector, timeout=timeout * 1000)
            return True
        except PlaywrightTimeout:
            return False
    
    def click(self, selector: str, timeout: float = 5.0) -> bool:
        """
        Click element matching selector.
        
        Returns:
            True if successful, False if failed
        """
        if not self.page:
            return False
        
        try:
            self.page.click(selector, timeout=timeout * 1000)
            return True
        except Exception as e:
            return False
    
    def type_text(self, selector: str, text: str, delay: float = 0.05) -> bool:
        """
        Type text into element.
        
        Args:
            selector: Element selector
            text: Text to type
            delay: Delay between keystrokes (seconds)
        
        Returns:
            True if successful, False if failed
        """
        if not self.page:
            return False
        
        try:
            self.page.fill(selector, text)
            return True
        except Exception:
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a key (Enter, Escape, etc.)."""
        if not self.page:
            return False
        
        try:
            self.page.keyboard.press(key)
            return True
        except Exception:
            return False
    
    def screenshot(self, path: Path):
        """Take a screenshot."""
        if not self.page:
            return
        
        self.page.screenshot(path=str(path))
    
    def close(self):
        """Close browser and cleanup."""
        if self.page:
            self.page.close()
            self.page = None
        
        if self.browser:
            self.browser.close()
            self.browser = None
        
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()