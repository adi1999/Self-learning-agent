"""Browser element capture using Playwright with persistent profile."""
from playwright.sync_api import sync_playwright, Page, Browser, Playwright, BrowserContext
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from src.models.session_artifact import ElementInfo
from src.utils.logger import setup_logger
from src.utils.config import config


class BrowserCapture:
    """
    Captures browser state and element information using Playwright.
    
    Now uses persistent profile (same as executor) to:
    - Share login sessions between recording and replay
    - Reduce CAPTCHA triggers
    - Provide consistent browser state
    """
    
    # JavaScript to capture element info
    ELEMENT_CAPTURE_JS = """
    (coords) => {
        const [x, y] = coords;
        const el = document.elementFromPoint(x, y);
        if (!el) return null;
        
        // Compute a robust selector
        function computeSelector(element) {
            if (element.id) return '#' + element.id;
            if (element.name) return element.tagName.toLowerCase() + '[name="' + element.name + '"]';
            
            // Try aria-label
            const ariaLabel = element.getAttribute('aria-label');
            if (ariaLabel) return `[aria-label="${ariaLabel}"]`;
            
            // Try data-testid
            const testId = element.getAttribute('data-testid');
            if (testId) return `[data-testid="${testId}"]`;
            
            // Build path-based selector
            let path = [];
            let current = element;
            while (current && current.nodeType === Node.ELEMENT_NODE) {
                let selector = current.tagName.toLowerCase();
                if (current.className && typeof current.className === 'string') {
                    const classes = current.className.trim().split(/\\s+/).slice(0, 2);
                    if (classes.length > 0 && classes[0]) {
                        selector += '.' + classes.join('.');
                    }
                }
                path.unshift(selector);
                current = current.parentElement;
                if (path.length >= 3) break;
            }
            return path.join(' > ');
        }
        
        // Get input type for input elements
        let inputType = null;
        if (el.tagName === 'INPUT') {
            inputType = el.type || 'text';
        }
        
        // Check if it's a search input
        const isSearchInput = (
            inputType === 'search' ||
            el.name === 'q' ||
            el.getAttribute('aria-label')?.toLowerCase().includes('search') ||
            el.placeholder?.toLowerCase().includes('search')
        );
        
        return {
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            classes: [...el.classList],
            name: el.getAttribute('name'),
            type: inputType,
            role: el.getAttribute('role'),
            ariaLabel: el.getAttribute('aria-label'),
            placeholder: el.placeholder || null,
            text: el.innerText?.substring(0, 100) || null,
            href: el.href || null,
            selector: computeSelector(el),
            isContentEditable: el.isContentEditable,
            isSearchInput: isSearchInput,
            value: el.value || null,
            rect: el.getBoundingClientRect().toJSON()
        };
    }
    """
    
    def __init__(self):
        self.logger = setup_logger("BrowserCapture")
        
        self.playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Use SAME profile directory as executor for session sharing
        self.profile_dir = Path.home() / ".pbd-browser-profile"
    
    def launch(
        self, 
        url: Optional[str] = None, 
        headless: bool = False,
        use_persistent_profile: bool = True
    ) -> Page:
        """
        Launch browser and return page.
        
        Now uses persistent profile by default to share sessions
        with the executor (replay). This means:
        - Login once during recording or replay
        - Session is shared across both
        - Reduces CAPTCHA triggers
        
        Args:
            url: Initial URL to navigate to
            headless: Run in headless mode
            use_persistent_profile: Use persistent browser profile (default: True)
        
        Returns:
            Playwright Page object
        """
        self.playwright = sync_playwright().start()
        
        if use_persistent_profile:
            # Ensure profile directory exists
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Using persistent profile: {self.profile_dir}")
            
            # Launch with persistent context (same as executor)
            self.context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=headless,
                channel="chrome",  # Use system Chrome if available
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="America/Los_Angeles",
                # More human-like settings
                java_script_enabled=True,
                bypass_csp=False,
                ignore_https_errors=False,
                # Reduce bot detection
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ]
            )
            
            # Use existing page or create new one
            if self.context.pages:
                self.page = self.context.pages[0]
            else:
                self.page = self.context.new_page()
        else:
            # Standard non-persistent launch (legacy behavior)
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                channel="chrome"
            )
            self.context = self.browser.new_context(
                viewport={"width": 1280, "height": 800}
            )
            self.page = self.context.new_page()
        
        # Set up event listeners
        self._setup_listeners()
        
        # Navigate to initial URL
        if url:
            self.page.goto(url, wait_until="domcontentloaded")
            self.logger.info(f"Browser launched at: {url}")
        else:
            self.logger.info("Browser launched (persistent profile)")
        
        return self.page
    
    def _setup_listeners(self):
        """Set up page event listeners."""
        if not self.page:
            return
        
        # Listen for console messages (debugging)
        self.page.on("console", lambda msg: 
            self.logger.debug(f"Browser console: {msg.text}")
            if msg.type == "error" else None
        )
        
        # Listen for page crashes
        self.page.on("crash", lambda: 
            self.logger.error("Browser page crashed!")
        )
    
    def get_element_at_point(self, x: int, y: int) -> Optional[ElementInfo]:
        """
        Get element information at screen coordinates.
        
        Note: Coordinates should be relative to the page viewport,
        not absolute screen coordinates.
        """
        if not self.page:
            return None
        
        try:
            result = self.page.evaluate(self.ELEMENT_CAPTURE_JS, [x, y])
            
            if not result:
                return None
            
            return ElementInfo(
                tag=result.get("tag"),
                element_id=result.get("id"),
                classes=result.get("classes"),
                name=result.get("name"),
                input_type=result.get("type"),
                role=result.get("role"),
                aria_label=result.get("ariaLabel"),
                placeholder=result.get("placeholder"),
                text=result.get("text"),
                href=result.get("href"),
                selector=result.get("selector"),
                is_content_editable=result.get("isContentEditable")
            )
        
        except Exception as e:
            self.logger.error(f"Failed to get element at point: {e}")
            return None
    
    def get_focused_element(self) -> Optional[ElementInfo]:
        """Get information about the currently focused element."""
        if not self.page:
            return None
        
        js = """
        () => {
            const el = document.activeElement;
            if (!el || el === document.body) return null;
            
            return {
                tag: el.tagName.toLowerCase(),
                id: el.id || null,
                name: el.getAttribute('name'),
                type: el.type || null,
                role: el.getAttribute('role'),
                ariaLabel: el.getAttribute('aria-label'),
                placeholder: el.placeholder || null,
                isContentEditable: el.isContentEditable,
                value: el.value || null
            };
        }
        """
        
        try:
            result = self.page.evaluate(js)
            if result:
                return ElementInfo(
                    tag=result.get("tag"),
                    element_id=result.get("id"),
                    name=result.get("name"),
                    input_type=result.get("type"),
                    role=result.get("role"),
                    aria_label=result.get("ariaLabel"),
                    placeholder=result.get("placeholder"),
                    is_content_editable=result.get("isContentEditable")
                )
        except Exception as e:
            self.logger.debug(f"Failed to get focused element: {e}")
        
        return None
    
    def get_current_url(self) -> str:
        """Get current page URL."""
        if not self.page:
            return ""
        return self.page.url
    
    def get_page_title(self) -> str:
        """Get current page title."""
        if not self.page:
            return ""
        return self.page.title()
    
    def get_page_state(self) -> Dict[str, Any]:
        """Get current page state."""
        return {
            "url": self.get_current_url(),
            "title": self.get_page_title()
        }
    
    def take_screenshot(self, path: Path) -> Optional[Path]:
        """Take a screenshot of the current page."""
        if not self.page:
            return None
        
        try:
            self.page.screenshot(path=str(path))
            return path
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            return None
    
    def wait_for_navigation(self, timeout: float = 10.0) -> bool:
        """Wait for page navigation to complete."""
        if not self.page:
            return False
        
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
            return True
        except Exception:
            return False
    
    def is_single_line_input(self, element_info: Optional[ElementInfo]) -> bool:
        """Check if element is a single-line input (Enter = submit)."""
        if not element_info:
            return False
        
        if element_info.tag == "textarea":
            return False
        
        if element_info.is_content_editable:
            return False
        
        if element_info.tag == "input":
            multi_line_types = ["textarea"]
            if element_info.input_type not in multi_line_types:
                return True
        
        return False
    
    def screen_to_viewport_coords(self, screen_x: int, screen_y: int) -> tuple:
        """Convert screen coordinates to viewport coordinates."""
        if not self.page:
            return (screen_x, screen_y)
        
        try:
            js = """
            () => ({
                screenX: window.screenX,
                screenY: window.screenY,
                outerWidth: window.outerWidth,
                outerHeight: window.outerHeight,
                innerWidth: window.innerWidth,
                innerHeight: window.innerHeight
            })
            """
            bounds = self.page.evaluate(js)
            
            chrome_x = (bounds["outerWidth"] - bounds["innerWidth"]) // 2
            chrome_y = bounds["outerHeight"] - bounds["innerHeight"] - chrome_x
            
            viewport_x = screen_x - bounds["screenX"] - chrome_x
            viewport_y = screen_y - bounds["screenY"] - chrome_y
            
            return (viewport_x, viewport_y)
        
        except Exception as e:
            self.logger.debug(f"Coord conversion failed: {e}")
            return (screen_x, screen_y)
    
    def close(self):
        """Close browser and cleanup."""
        import time
        time.sleep(0.2)  # Let pending operations complete
        
        if self.page:
            try:
                self.page.close()
            except:
                pass
            self.page = None
        
        if self.context:
            try:
                self.context.close()
            except:
                pass
            self.context = None
        
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
            self.browser = None
        
        if self.playwright:
            try:
                self.playwright.stop()
            except:
                pass
            self.playwright = None
        
        self.logger.info("Browser closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()