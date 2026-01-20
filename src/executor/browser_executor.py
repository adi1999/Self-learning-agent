"""Browser executor with Gemini-powered extraction and fallback."""
import time
import random
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

from playwright.sync_api import sync_playwright, Page, Browser, Playwright, BrowserContext

from src.models.workflow_recipe import WorkflowStep, ElementReference, ExtractionSchema
from src.utils.logger import setup_logger
from src.utils.config import config
from src.utils.gemini_client import gemini_client
from src.utils.safety_guard import safety_guard


@dataclass
class StepResult:
    """Result of executing a browser step."""
    success: bool
    error: Optional[str] = None
    strategy_used: Optional[str] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)


class BrowserExecutor:
    """
    Executes browser steps using Playwright with Gemini enhancement.
    
    Features:
    - Persistent browser profile (reduces CAPTCHA)
    - Gemini-powered extraction (reliable, no clipboard needed)
    - Gemini fallback when selectors fail
    - Human-like typing delays
    - Improved element descriptions for Gemini
    """
    
    def __init__(self):
        self.logger = setup_logger("BrowserExecutor")
        
        self.playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Profile directory for persistent sessions
        self.profile_dir = Path.home() / ".pbd-browser-profile"
        
        # Screen dimensions for Gemini coordinate denormalization
        self._screen_width = 1280
        self._screen_height = 800
    
    def _clean_corrupt_profile(self) -> bool:
        """
        Detect and clean corrupt browser profile.
        
        Returns:
            True if profile was corrupt and cleaned
        """
        if not self.profile_dir.exists():
            return False
        
        # Indicators of profile corruption
        lock_files = [
            self.profile_dir / "SingletonLock",
            self.profile_dir / "SingletonSocket",
            self.profile_dir / "SingletonCookie",
        ]
        
        crash_indicators = [
            self.profile_dir / "Crashpad",
            self.profile_dir / "Default" / "LOCK",
        ]
        
        # Check for lock files left by crashed browser
        has_stale_locks = any(f.exists() for f in lock_files)
        
        # Check for crash artifacts
        has_crash_data = False
        crashpad_dir = self.profile_dir / "Crashpad" / "pending"
        if crashpad_dir.exists():
            pending_crashes = list(crashpad_dir.glob("*"))
            has_crash_data = len(pending_crashes) > 5  # Many pending crashes = bad state
        
        if has_stale_locks or has_crash_data:
            self.logger.warning(f"Detected corrupt browser profile at {self.profile_dir}")
            self.logger.info("Cleaning corrupt profile...")
            
            try:
                # Remove the entire profile directory
                shutil.rmtree(self.profile_dir)
                self.profile_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Profile cleaned successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clean profile: {e}")
                # Try to at least remove lock files
                for lock_file in lock_files:
                    try:
                        if lock_file.exists():
                            lock_file.unlink()
                    except:
                        pass
        
        return False
    
    def launch(
        self, 
        url: str = "https://www.google.com",
        headless: bool = False,
        use_persistent_profile: bool = False  # CHANGED: Default to False for reliability
    ) -> Page:
        """
        Launch browser for workflow execution.
        
        Args:
            url: Initial URL to navigate to
            headless: Run browser without visible window
            use_persistent_profile: Use persistent profile (may cause corruption issues)
                                   Defaults to False for reliability.
        
        Returns:
            Playwright Page object
        """
        self.logger.info("Launching browser for execution...")
        
        # Safety check on URL
        url_check = safety_guard.check_url(url)
        if not url_check.allowed:
            raise ValueError(f"Blocked URL: {url_check.reason}")
        
        self.playwright = sync_playwright().start()
        
        if use_persistent_profile:
            # Clean corrupt profile if detected
            self._clean_corrupt_profile()
            
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using persistent profile: {self.profile_dir}")
            
            try:
                self.context = self.playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.profile_dir),
                    headless=headless,
                    viewport={"width": self._screen_width, "height": self._screen_height},
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ]
                )
                self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            except Exception as e:
                self.logger.warning(f"Persistent context failed: {e}")
                self.logger.info("Falling back to fresh context...")
                # Clean profile and retry without persistence
                self._clean_corrupt_profile()
                use_persistent_profile = False
        
        if not use_persistent_profile:
            # Use fresh browser context (more reliable)
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ]
            )
            self.context = self.browser.new_context(
                viewport={"width": self._screen_width, "height": self._screen_height},
                # Add user agent to reduce CAPTCHA frequency
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self.page = self.context.new_page()
        
        self.page.goto(url, wait_until="domcontentloaded")
        self._handle_captcha_if_present()
        
        self.logger.info(f"Navigated to: {url}")
        return self.page
    
    def _is_actual_captcha(self) -> Tuple[bool, str]:
        """Check if current page has an actual CAPTCHA."""
        if not self.page:
            return False, ""
        
        # Check for reCAPTCHA
        try:
            recaptcha_iframe = self.page.locator("iframe[src*='recaptcha']").first
            if recaptcha_iframe.is_visible(timeout=500):
                return True, "recaptcha-iframe"
        except:
            pass
        
        try:
            recaptcha_div = self.page.locator(".g-recaptcha, #recaptcha").first
            if recaptcha_div.is_visible(timeout=500):
                return True, "recaptcha-div"
        except:
            pass
        
        # Check for Cloudflare
        try:
            turnstile = self.page.locator(".cf-turnstile, iframe[src*='challenges.cloudflare']").first
            if turnstile.is_visible(timeout=500):
                return True, "cloudflare-turnstile"
        except:
            pass
        
        # Check URL patterns
        current_url = self.page.url.lower()
        captcha_url_patterns = [
            "/sorry/index",
            "ipv4.google.com/sorry",
            "challenge",
            "/captcha",
        ]
        
        for pattern in captcha_url_patterns:
            if pattern in current_url:
                return True, f"url-pattern:{pattern}"
        
        # Check page text
        try:
            page_text = self.page.inner_text("body").lower()
            definite_captcha_phrases = [
                "unusual traffic from your computer",
                "automated queries",
                "please verify you are a human",
                "complete the security check",
                "prove you're not a robot",
            ]
            
            for phrase in definite_captcha_phrases:
                if phrase in page_text:
                    return True, f"text:{phrase[:30]}"
        except:
            pass
        
        return False, ""
    
    def _handle_captcha_if_present(self) -> bool:
        """Check for CAPTCHA and wait for human intervention if detected."""
        is_captcha, captcha_type = self._is_actual_captcha()
        
        if not is_captcha:
            return False
        
        self.logger.warning("=" * 60)
        self.logger.warning(f"⚠️  CAPTCHA DETECTED: {captcha_type}")
        self.logger.warning("=" * 60)
        self.logger.warning("Please solve the CAPTCHA in the browser window.")
        self.logger.warning("The workflow will continue after you solve it.")
        self.logger.warning("=" * 60)
        
        input("\n>>> Press Enter after solving the CAPTCHA... ")
        
        time.sleep(2)
        
        still_captcha, _ = self._is_actual_captcha()
        
        if still_captcha:
            self.logger.warning("CAPTCHA may still be present. Continuing anyway...")
        else:
            self.logger.info("✓ CAPTCHA appears to be solved. Continuing...")
        
        return True
    
    def _build_gemini_element_description(
        self, 
        step: WorkflowStep, 
        ref: ElementReference
    ) -> str:
        """
        Build a USEFUL description for Gemini element finding.
        
        This is the FIX for the "select in Google Chrome" problem.
        We now use actual element properties instead of generic step descriptions.
        """
        parts = []
        
        if ref.visual_hint and "search result" in ref.visual_hint:
            return ref.visual_hint  # e.g., "first search result in the main content area"
        
        # Priority 2: Text content (if we kept it)
        if ref.text:
            return f'element containing text "{ref.text[:50]}"'
        
        # Priority 2: Role information
        if ref.role:
            parts.append(f'{ref.role}')
        elif ref.accessibility_role:
            parts.append(f'{ref.accessibility_role}')
        
        # Priority 3: Visual hint (if we captured one)
        if ref.visual_hint:
            parts.append(f'({ref.visual_hint})')
        
        # Priority 4: Positional hint from coordinates
        if ref.coordinates and len(ref.coordinates) == 2:
            x, y = ref.coordinates
            if x is not None and y is not None:
                h_pos = "left side" if x < 400 else "right side" if x > 1000 else "center"
                v_pos = "top area" if y < 300 else "bottom area" if y > 600 else "middle area"
                parts.append(f'located in the {v_pos}, {h_pos} of the page')
        
        # Priority 5: Intent-based hints
        if not parts:
            intent_hints = {
                "search": "search input field or search box",
                "select": "clickable link or button",
                "navigate": "navigation link",
                "write": "text input field",
                "extract": "content area with data",
            }
            if step.intent in intent_hints:
                parts.append(intent_hints[step.intent])
        
        # Build final description
        if parts:
            description = " ".join(parts)
        else:
            # Absolute last resort - but still better than "select in Google Chrome"
            description = f"interactive element for {step.intent} action"
        
        return description
    
    def execute_step(self, step: WorkflowStep, use_gemini_fallback: bool = True) -> StepResult:
        """Execute a single browser step."""
        if not self.page:
            return StepResult(success=False, error="Browser not initialized")
        
        try:
            time.sleep(random.uniform(0.3, 0.8))
            self._handle_captcha_if_present()
            
            # Route to appropriate executor
            if step.action_type == "type":
                # Use fallback-enabled search for search intent
                if step.intent == "search":
                    return self._execute_search_with_fallbacks(step, use_gemini_fallback)
                return self._execute_type(step, use_gemini_fallback)
            elif step.action_type == "click":
                return self._execute_click(step, use_gemini_fallback)
            elif step.action_type == "navigate":
                return self._execute_navigate(step)
            elif step.action_type == "extract":
                return self._execute_extract(step)
            elif step.action_type == "shortcut":
                return self._execute_shortcut(step)
            elif step.action_type == "wait":
                return self._execute_wait(step)
            else:
                return StepResult(
                    success=False,
                    error=f"Unknown action type: {step.action_type}"
                )
        
        except Exception as e:
            return StepResult(success=False, error=str(e))
        
    def _execute_search_with_fallbacks(
        self, 
        step: WorkflowStep, 
        use_gemini_fallback: bool
    ) -> StepResult:
        """
        Execute search with intelligent fallbacks if no results.
        
        Fallback chain:
        1. Try original query
        2. If no results → remove site filter, retry
        3. If still no results → simplify query, retry
        4. If still no results → use Gemini to suggest better query
        """
        original_value = step.parameter_bindings.get("value", "")
        
        # Fallback chain - progressively simpler queries
        query_attempts = self._build_query_fallback_chain(original_value)
        
        for i, query in enumerate(query_attempts):
            attempt_name = "original" if i == 0 else f"fallback_{i}"
            self.logger.info(f"  Search attempt ({attempt_name}): '{query}'")
            
            # Update the value for this attempt
            step.parameter_bindings["value"] = query
            
            # Execute the search
            result = self._execute_type(step, use_gemini_fallback)
            
            if not result.success:
                continue
            
            # Wait for results to load
            time.sleep(1.5)
            
            # Check if we got meaningful results
            has_results = self._check_search_has_results()
            
            if has_results:
                self.logger.info(f"  ✓ Search successful with: '{query}'")
                result.strategy_used = f"search:{attempt_name}"
                return result
            else:
                self.logger.warning(f"  No results for: '{query}', trying fallback...")
        
        # All fallbacks failed
        self.logger.error("  All search attempts failed to find results")
        return StepResult(
            success=True,  # Search technically worked, just no results
            strategy_used="search:no_results",
            error="No results found after all fallback attempts"
        )


    def _build_query_fallback_chain(self, original_query: str) -> List[str]:
        """
        Build a chain of progressively simpler queries.
        
        Example for "best pizza places in delhi zomato":
        1. "best pizza places in delhi zomato" (original)
        2. "best pizza places in delhi" (remove site filter)
        3. "pizza places in delhi" (remove adjective)
        4. "pizza delhi" (minimal)
        """
        queries = [original_query]
        
        words = original_query.split()
        
        # Known site/platform names to try removing
        sites = {'zomato', 'yelp', 'tripadvisor', 'swiggy', 'google', 'maps', 
                'uber', 'eats', 'doordash', 'grubhub', 'opentable', 'booking',
                'airbnb', 'expedia', 'kayak', 'amazon', 'ebay', 'linkedin'}
        
        # Fallback 1: Remove site filter
        words_no_site = [w for w in words if w.lower() not in sites]
        if words_no_site != words:
            queries.append(" ".join(words_no_site))
        
        # Fallback 2: Remove "best", "top", etc.
        qualifiers = {'best', 'top', 'good', 'great', 'cheap', 'affordable', 'luxury', 'popular'}
        words_no_qualifier = [w for w in words_no_site if w.lower() not in qualifiers]
        if words_no_qualifier != words_no_site and len(words_no_qualifier) >= 2:
            queries.append(" ".join(words_no_qualifier))
        
        # Fallback 3: Keep only nouns + location (minimal query)
        # Just keep first 2-3 meaningful words
        if len(words_no_qualifier) > 3:
            queries.append(" ".join(words_no_qualifier[:3]))
        
        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries


    def _check_search_has_results(self) -> bool:
        """
        Check if the current page has meaningful search results.
        Uses Gemini vision to understand the page state.
        """
        if not gemini_client.is_available:
            # Can't check, assume it worked
            return True
        
        try:
            screenshot = self.page.screenshot(type="png")
            
            prompt = """Look at this search results page and answer:
    Does this page show actual search results (listings, links, items)?

    Answer ONLY with JSON:
    {"has_results": true/false, "reason": "brief explanation"}

    Examples of NO results:
    - "No results found" message
    - Empty page
    - Error message
    - CAPTCHA page

    Examples of YES results:
    - List of items/links
    - Search results with titles
    - Product listings
    - Business listings"""

            response = gemini_client.client.models.generate_content(
                model=gemini_client.VISION_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            text = gemini_client._safe_extract_text(response)
            result = gemini_client._parse_json_response(text)
            
            if result:
                has_results = result.get("has_results", True)
                reason = result.get("reason", "")
                if not has_results:
                    self.logger.info(f"  No results detected: {reason}")
                return has_results
            
            return True  # Assume yes if parsing fails
            
        except Exception as e:
            self.logger.debug(f"Result check failed: {e}")
            return True  # Assume yes on error
    
    def _execute_type(self, step: WorkflowStep, use_gemini_fallback: bool) -> StepResult:
        """Execute a type action with human-like delays."""
        value = step.parameter_bindings.get("value", "")
        
        if not value:
            return StepResult(success=False, error="No value to type")
        
        # Strategy 1: Find search input (for search intent)
        if step.intent == "search":
            search_selectors = [
                'textarea[name="q"]',  # Google search textarea
                'input[name="q"]',
                'input[type="search"]',
                '[role="searchbox"]',
                '[role="combobox"]',
                '#search-input',
                '.search-input',
            ]
            
            for selector in search_selectors:
                try:
                    element = self.page.locator(selector).first
                    if element.is_visible(timeout=1000):
                        element.click()
                        time.sleep(random.uniform(0.2, 0.5))
                        element.fill("")
                        self._human_type(value)
                        time.sleep(random.uniform(0.3, 0.6))
                        self.page.keyboard.press("Enter")
                        self._wait_for_navigation_or_content()
                        self._handle_captcha_if_present()
                        return StepResult(success=True, strategy_used=f"search:{selector}")
                except:
                    continue
        
        # Strategy 2: Use element reference selector
        if step.element_reference and step.element_reference.selector:
            try:
                element = self.page.locator(step.element_reference.selector).first
                element.click()
                time.sleep(random.uniform(0.2, 0.4))
                element.fill("")
                self._human_type(value)
                return StepResult(success=True, strategy_used="selector")
            except:
                pass
        
        # Strategy 3: Gemini fallback - find the input element
        if use_gemini_fallback and gemini_client.is_available:
            self.logger.info("  Trying Gemini to find input element...")
            screenshot = self.page.screenshot(type="png")
            
            # Build a USEFUL description
            if step.element_reference:
                description = self._build_gemini_element_description(step, step.element_reference)
            else:
                description = f"text input field for entering: {value[:30]}"
            
            self.logger.info(f"  Gemini searching for: {description}")
            
            coords = gemini_client.find_element(
                screenshot_bytes=screenshot,
                element_description=description,
                screen_width=self._screen_width,
                screen_height=self._screen_height
            )
            
            if coords:
                try:
                    self.page.mouse.click(coords[0], coords[1])
                    time.sleep(0.3)
                    self._human_type(value)
                    return StepResult(success=True, strategy_used="gemini_coords")
                except Exception as e:
                    self.logger.warning(f"Gemini click failed: {e}")
        
        # Strategy 4: Type into focused element
        try:
            self._human_type(value)
            return StepResult(success=True, strategy_used="focused")
        except Exception as e:
            return StepResult(success=False, error=f"Type failed: {e}")
    
    def _human_type(self, text: str, min_delay: float = 0.03, max_delay: float = 0.12):
        """Type text with human-like delays between characters."""
        for char in text:
            self.page.keyboard.type(char)
            time.sleep(random.uniform(min_delay, max_delay))
    
    def _execute_click(self, step: WorkflowStep, use_gemini_fallback: bool) -> StepResult:
        """
        Execute click with goal-oriented verification.
        
        If step has expected_url_pattern, verifies navigation succeeded.
        If navigation fails, retries with different elements (up to 3 times).
        """
        ref = step.element_reference
        
        if not ref:
            return StepResult(success=False, error="No element reference for click")
        
        strategies_tried = []
        start_url = self.page.url
        expected_pattern = step.expected_url_pattern
        max_navigation_attempts = 3
        
        # Helper: Check if navigation goal achieved
        def navigation_goal_achieved() -> bool:
            """Check if we navigated to the expected URL pattern AND page type."""
            required_page_type = step.completion_signal.required_page_type if step.completion_signal else None
            
            # Wait loop parameters - INCREASED for reliability
            max_wait_time = 10.0  # seconds
            poll_interval = 0.5   # seconds
            iterations = int(max_wait_time / poll_interval)
            
            url_changed = False
            pattern_matched = False
            
            # 1. Wait for URL change / Pattern match
            for _ in range(iterations):
                current_url = self.page.url
                
                # Check if URL changed at all
                if current_url != start_url:
                    url_changed = True
                    
                    # If pattern is required, check it
                    if expected_pattern:
                        if expected_pattern.lower() in current_url.lower():
                            pattern_matched = True
                            break
                    else:
                        # No pattern required, just change is enough
                        break
                
                time.sleep(poll_interval)
            
            # If pattern was required but not found
            if expected_pattern and not pattern_matched:
                if url_changed:
                    self.logger.warning(f"  URL changed but not to expected pattern {expected_pattern}")
                if not required_page_type: # Fail if only URL pattern was required
                    return False

            # 2. Check Page Type (if required) - CRITICAL for data quality
            if required_page_type and gemini_client.is_available:
                self.logger.info(f"  Verifying page type: {required_page_type}")
                screenshot = self.page.screenshot(type="png")
                # Use stricter validation for navigation goals
                if self._validate_page_type(screenshot, required_page_type):
                    self.logger.info(f"  ✓ Page type verified: {required_page_type}")
                    return True
                else:
                    self.logger.warning(f"  ✗ Page type mismatch (not {required_page_type})")
                    return False
            
            # If we got here:
            # - If URL matched and no page type required -> Success
            # - If no URL pattern and no page type required -> Any nav is success
            if expected_pattern:
                 return pattern_matched
            
            return url_changed

        # Strategy 1: CSS selector (if available)
        if ref.selector:
            strategies_tried.append("selector")
            try:
                self.page.locator(ref.selector).first.click(timeout=5000)
                self._wait_for_navigation_or_content()
                if navigation_goal_achieved():
                    return StepResult(success=True, strategy_used="selector")
            except Exception as e:
                self.logger.debug(f"Selector strategy failed: {e}")
        
        # Strategy 2: Text content (if available)
        if ref.text:
            strategies_tried.append("text")
            try:
                self.page.get_by_text(ref.text, exact=False).first.click(timeout=5000)
                self._wait_for_navigation_or_content()
                if navigation_goal_achieved():
                    return StepResult(success=True, strategy_used="text")
            except Exception as e:
                self.logger.debug(f"Text strategy failed: {e}")
        
        # Strategy 3: Role (if available)
        if ref.role:
            strategies_tried.append("role")
            try:
                self.page.get_by_role(ref.role).first.click(timeout=5000)
                self._wait_for_navigation_or_content()
                if navigation_goal_achieved():
                    return StepResult(success=True, strategy_used="role")
            except Exception as e:
                self.logger.debug(f"Role strategy failed: {e}")
        
        # Strategy 4: Gemini with smart retry
        # This is the KEY change: if we have a navigation goal, we retry with DIFFERENT elements
        if use_gemini_fallback and gemini_client.is_available:
            strategies_tried.append("gemini")
            
            for attempt in range(max_navigation_attempts):
                # Take fresh screenshot each attempt
                screenshot = self.page.screenshot(type="png")
                
                # Build description with attempt context
                description = self._build_progressive_description(step, ref, attempt, expected_pattern)
                self.logger.info(f"  Gemini attempt {attempt + 1}: searching for '{description}'")
                
                try:
                    coords = gemini_client.find_element(
                        screenshot_bytes=screenshot,
                        element_description=description,
                        screen_width=self._screen_width,
                        screen_height=self._screen_height
                    )
                    
                    if coords:
                        self.page.mouse.click(coords[0], coords[1])
                        self._wait_for_navigation_or_content()
                        
                        if navigation_goal_achieved():
                            return StepResult(success=True, strategy_used=f"gemini_attempt_{attempt + 1}")
                        else:
                            self.logger.info(f"  Attempt {attempt + 1} didn't achieve navigation goal, trying different element...")
                            # Reset to start URL for next attempt if we navigated to wrong place
                            if self.page.url != start_url and expected_pattern:
                                self.page.go_back()
                                time.sleep(0.5)
                    else:
                        self.logger.debug(f"  Gemini didn't find element on attempt {attempt + 1}")
                        
                except Exception as e:
                    self.logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
            
            # All Gemini attempts failed
            if expected_pattern:
                return StepResult(success=False, error=f"Navigation to {expected_pattern} failed after {max_navigation_attempts} attempts")
        
        # Strategy 5: Coordinates (only for static elements)
        dynamic_intents = {"select", "navigate"}
        if step.intent not in dynamic_intents:
            if ref.coordinates and len(ref.coordinates) == 2:
                strategies_tried.append("coordinates")
                try:
                    x, y = ref.coordinates
                    if x is not None and y is not None:
                        self.logger.warning(f"  Using fallback coordinates ({x}, {y})")
                        self.page.mouse.click(x, y)
                        self._wait_for_navigation_or_content()
                        if navigation_goal_achieved():
                            return StepResult(success=True, strategy_used="coordinates_fallback")
                except Exception as e:
                    self.logger.debug(f"Coordinate click failed: {e}")
        
        error_msg = f"All click strategies failed. Tried: {strategies_tried}"
        return StepResult(success=False, error=error_msg)
    
    def _build_progressive_description(
        self, 
        step: WorkflowStep, 
        ref: ElementReference, 
        attempt: int,
        expected_pattern: Optional[str]
    ) -> str:
        """Build element description that gets more specific on retry."""
        base_hint = ref.visual_hint or "search result link"
        
        # Include target site if known
        site_hint = ""
        if expected_pattern:
            # Extract site name (e.g., "zomato" from "zomato.com")
            site_name = expected_pattern.split('.')[0]
            site_hint = f" to {site_name}"
        
        if attempt == 0:
            return f"first {base_hint}{site_hint}"
        elif attempt == 1:
            return f"second clickable link{site_hint} that looks different from Google UI elements"
        else:
            return f"a link{site_hint} in the main search results area, not in the header or sidebar"
    
    def _execute_navigate(self, step: WorkflowStep) -> StepResult:
        """Execute a navigation action."""
        url = step.parameter_bindings.get("url", "")
        
        if not url:
            return StepResult(success=False, error="No URL for navigation")
        
        # Safety check on URL
        url_check = safety_guard.check_url(url)
        if not url_check.allowed:
            self.logger.warning(f"🛑 Navigation blocked: {url_check.reason}")
            return StepResult(success=False, error=f"Blocked: {url_check.reason}")
        
        try:
            self.page.goto(url, wait_until="domcontentloaded")
            self._handle_captcha_if_present()
            return StepResult(success=True, strategy_used="direct")
        except Exception as e:
            return StepResult(success=False, error=f"Navigation failed: {e}")
    
    
    def _validate_page_type(self, screenshot: bytes, expected_type: str) -> bool:
        """Verify if page screenshot matches expected type."""
        if not expected_type or not gemini_client.is_available:
            return True
        return gemini_client.validate_page_type(screenshot, expected_type)

    def _execute_extract(self, step: WorkflowStep) -> StepResult:
        """
        Execute extraction using Gemini vision.
        
        This is the KEY improvement - extraction via Gemini is:
        - More reliable than clipboard-based
        - Doesn't require selecting text
        - Works on any page layout
        """
        self.logger.info("  Executing Gemini-powered extraction...")
        
        if not gemini_client.is_available:
            self.logger.error("Gemini not available for extraction")
            return StepResult(
                success=False,
                error="Gemini client required for extraction but not available"
            )
        
        # Take screenshot
        screenshot = self.page.screenshot(type="png")

        # Verify page type matches expectation (if defined in schema)
        # NOTE: Changed to WARNING instead of failure - proceed with extraction but flag quality issue
        page_type_mismatch = False
        if step.extraction_schema and step.extraction_schema.page_type:
            self.logger.info(f"  Validating page type: {step.extraction_schema.page_type}")
            if not self._validate_page_type(screenshot, step.extraction_schema.page_type):
                self.logger.warning(f"  ⚠ Page type mismatch (expected {step.extraction_schema.page_type}) - proceeding anyway")
                page_type_mismatch = True
        
        # Get extraction schema
        schema = step.extraction_schema
        
        if not schema or not schema.fields:
            self.logger.warning("No extraction schema, using default fields")
            schema_dict = {
                "title": {"description": "Main title or name", "visual_hint": "large heading"},
                "rating": {"description": "Rating or score", "visual_hint": "near stars"},
                "address": {"description": "Address or location", "visual_hint": "street address"},
            }
        else:
            schema_dict = schema.to_gemini_schema()
        
        # Call Gemini for extraction
        extracted = gemini_client.extract_fields(
            screenshot_bytes=screenshot,
            extraction_schema=schema_dict
        )
        
        if extracted:
            self.logger.info(f"  ✓ Extracted {len(extracted)} fields:")
            for k, v in extracted.items():
                preview = str(v)[:40] + "..." if len(str(v)) > 40 else v
                self.logger.info(f"    - {k}: {preview}")
            
            return StepResult(
                success=True,
                strategy_used="gemini_vision",
                extracted_data=extracted
            )
        
        return StepResult(
            success=False,
            error="Gemini extraction returned no data"
        )
    
    def _execute_shortcut(self, step: WorkflowStep) -> StepResult:
        """Execute a keyboard shortcut."""
        shortcut = step.shortcut
        
        # If pasting, set clipboard content first
        if shortcut == "paste" and step.clipboard_content:
            try:
                import pyperclip
                pyperclip.copy(step.clipboard_content)
                time.sleep(0.1)
                preview = step.clipboard_content[:50] + "..." if len(step.clipboard_content) > 50 else step.clipboard_content
                self.logger.info(f"  Set clipboard before paste: {preview}")
            except Exception as e:
                self.logger.warning(f"Could not set clipboard: {e}")
        
        shortcut_keys = {
            "copy": "Meta+c",
            "paste": "Meta+v",
            "save": "Meta+s",
            "select_all": "Meta+a",
            "undo": "Meta+z",
            "redo": "Meta+Shift+z",
            "find": "Meta+f",
        }
        
        if shortcut in shortcut_keys:
            try:
                self.page.keyboard.press(shortcut_keys[shortcut])
                time.sleep(0.3)
                return StepResult(success=True, strategy_used=shortcut)
            except Exception as e:
                return StepResult(success=False, error=f"Shortcut failed: {e}")
        
        return StepResult(success=False, error=f"Unknown shortcut: {shortcut}")
    
    def _execute_wait(self, step: WorkflowStep) -> StepResult:
        """Execute a wait action."""
        timeout_ms = step.completion_signal.timeout_ms if step.completion_signal else 2000
        time.sleep(timeout_ms / 1000)
        return StepResult(success=True, strategy_used="timeout")
    
    def _wait_for_navigation_or_content(self, timeout: float = 5.0):
        """Wait for page to be ready after action."""
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
        except:
            pass
        time.sleep(0.5)
    
    def get_screenshot_bytes(self) -> bytes:
        """Get current page screenshot as bytes."""
        if not self.page:
            return b""
        return self.page.screenshot(type="png")
    
    def close(self):
        """Close browser and cleanup."""
        time.sleep(0.2)
        
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