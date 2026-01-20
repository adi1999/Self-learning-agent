"""Goal Executor - achieves goals using multiple strategies.

Key principle: Success is determined by CRITERIA, not by action completion.
We try strategies until the success criteria are met, then move on.
"""
import time
import random
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page

from src.models.goal_step import (
    GoalStep, GoalType, SuccessCriteria, Strategy, GoalWorkflow
)
from src.utils.logger import setup_logger
from src.utils.gemini_client import gemini_client
from src.utils.safety_guard import safety_guard


@dataclass
class GoalResult:
    """Result of attempting to achieve a goal."""
    achieved: bool
    strategy_used: Optional[str] = None
    attempts: int = 0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    fallback_used: bool = False
    duration_seconds: float = 0.0


@dataclass  
class WorkflowResult:
    """Result of executing a complete workflow."""
    success: bool
    steps_executed: int = 0
    steps_failed: int = 0
    total_steps: int = 0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    step_results: List[GoalResult] = field(default_factory=list)
    duration_seconds: float = 0.0


class GoalExecutor:
    """
    Executes goal-oriented steps by trying multiple strategies.
    
    Key differences from action-based execution:
    1. Success is determined by CRITERIA, not action completion
    2. Multiple strategies are tried until criteria are met
    3. Gemini agent loop as ultimate fallback
    """
    
    def __init__(
        self,
        browser_page: Optional[Page] = None,
        desktop_executor: Optional[Any] = None,
        app_launcher: Optional[Any] = None
    ):
        self.page = browser_page
        self.desktop_executor = desktop_executor
        self.app_launcher = app_launcher
        self.logger = setup_logger("GoalExecutor")
        
        # Screen dimensions for Gemini
        self._screen_width = 1280
        self._screen_height = 800
        
        # Extracted data store (flows between steps)
        self._extracted_data: Dict[str, Any] = {}
        self._last_extracted: Dict[str, Any] = {}  # Last extraction result
        
        # Current platform tracking
        self._current_platform: Optional[str] = None
        self._current_app: Optional[str] = None
    
    def set_browser_page(self, page: Page):
        """Set the browser page."""
        self.page = page
        viewport = page.viewport_size
        if viewport:
            self._screen_width = viewport.get("width", 1280)
            self._screen_height = viewport.get("height", 800)
    
    def execute_workflow(
        self,
        workflow: GoalWorkflow,
        parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute a complete goal-based workflow."""
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info(f"Executing Goal Workflow: {workflow.name}")
        self.logger.info("=" * 60)
        
        # Reset extracted data
        self._extracted_data = {}
        self._last_extracted = {}
        
        # Substitute parameters
        if parameters:
            self.logger.info(f"Parameters: {parameters}")
            workflow = workflow.substitute_parameters(parameters)
        
        self.logger.info(f"Steps: {len(workflow.steps)}")
        self.logger.info("=" * 60)
        
        result = WorkflowResult(
            success=True,
            total_steps=len(workflow.steps)
        )
        
        for i, goal_step in enumerate(workflow.steps):
            step_start = time.time()
            
            self.logger.info(f"\n[Step {i+1}/{len(workflow.steps)}] {goal_step.goal_description}")
            self.logger.info(f"  Goal: {goal_step.goal_type.value} | Platform: {goal_step.platform}")
            
            # Handle template filling for write/paste goals
            if goal_step.template:
                filled = workflow.fill_template(goal_step.template, self._extracted_data)
                goal_step.parameters["filled_template"] = filled
                self.logger.info(f"  Template filled: {filled[:50]}...")
            
            # Pass workflow parameters to goal parameters for context
            for key, value in (parameters or {}).items():
                if key not in goal_step.parameters:
                    goal_step.parameters[key] = value
            
            # Execute the goal
            goal_result = self.execute_goal(goal_step)
            goal_result.duration_seconds = time.time() - step_start
            
            result.step_results.append(goal_result)
            
            if goal_result.achieved:
                result.steps_executed += 1
                self.logger.info(f"  ✓ Goal achieved via {goal_result.strategy_used} ({goal_result.duration_seconds:.2f}s)")
                
                # Collect extracted data
                if goal_result.extracted_data:
                    self._extracted_data.update(goal_result.extracted_data)
                    # Also update _last_extracted for paste operations
                    self._last_extracted.update(goal_result.extracted_data)
                    self.logger.info(f"  ✓ Extracted {len(goal_result.extracted_data)} fields")
                    self.logger.info(f"  → Stored in _extracted_data: {len(self._extracted_data)} fields")
            else:
                result.steps_failed += 1
                result.errors.append(f"Step {i+1}: {goal_result.error}")
                self.logger.error(f"  ✗ Goal not achieved: {goal_result.error}")
                
                if not goal_step.optional:
                    self.logger.error("Aborting workflow due to failure")
                    result.success = False
                    break
        
        result.extracted_data = self._extracted_data
        result.duration_seconds = time.time() - start_time
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Workflow Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"Success: {result.success}")
        self.logger.info(f"Steps: {result.steps_executed}/{result.total_steps}")
        self.logger.info(f"Duration: {result.duration_seconds:.1f}s")
        
        if result.extracted_data:
            self.logger.info(f"\nExtracted Data:")
            self.logger.info("-" * 40)
            for field, value in result.extracted_data.items():
                # Truncate long values for display
                display_value = str(value)
                if len(display_value) > 80:
                    display_value = display_value[:80] + "..."
                self.logger.info(f"  {field}: {display_value}")
            self.logger.info("-" * 40)
        
        return result
    
    def execute_goal(self, goal: GoalStep) -> GoalResult:
        """
        Execute a single goal step using available strategies.
        
        Process:
        1. **NEW**: Pre-check if goal is already satisfied (state awareness)
        2. Handle platform switching
        3. Try each strategy in priority order
        4. After each attempt, check success criteria
        5. If all strategies fail and fallback enabled, use Gemini agent
        """
        result = GoalResult(achieved=False)
        start_url = self.page.url if self.page else ""
        
        # =========================================================================
        # STATE AWARENESS: Check if goal is already satisfied BEFORE trying
        # =========================================================================
        if self._is_goal_already_satisfied(goal, start_url):
            self.logger.info(f"  ✓ Goal already satisfied (state-aware skip)")
            result.achieved = True
            result.strategy_used = "already_satisfied"
            return result
        
        # Handle platform/app switching
        self._handle_platform_switch(goal)
        
        # Get strategies sorted by priority
        strategies = sorted(goal.strategies, key=lambda s: -s.priority)
        
        for retry in range(goal.max_retries):
            if retry > 0:
                self.logger.info(f"  Retry {retry}/{goal.max_retries}")
                time.sleep(1.0)
            
            for strategy in strategies:
                result.attempts += 1
                
                # Check platform compatibility
                if strategy.requires_platform and strategy.requires_platform != goal.platform:
                    continue
                
                self.logger.debug(f"  Trying: {strategy.name} (priority {strategy.priority})")
                
                try:
                    # Execute the strategy
                    self._execute_strategy(goal, strategy)
                    
                    # Wait for effects
                    time.sleep(goal.wait_after_seconds)
                    
                    # Check success criteria
                    if self._check_success_criteria(goal.success_criteria, goal, start_url):
                        result.achieved = True
                        result.strategy_used = strategy.name
                        
                        # Handle extraction
                        if goal.goal_type == GoalType.EXTRACT:
                            result.extracted_data = self._last_extracted or {}
                        
                        return result
                    else:
                        self.logger.debug(f"  Strategy {strategy.name}: executed but criteria not met")
                
                except Exception as e:
                    self.logger.debug(f"  Strategy {strategy.name} failed: {e}")
        
        # All strategies failed - try Gemini agent fallback
        if goal.fallback_to_agent and gemini_client.is_available and goal.agent_goal_prompt:
            self.logger.info("  Trying Gemini agent fallback...")
            
            if self._execute_agent_fallback(goal, start_url):
                result.achieved = True
                result.strategy_used = "gemini_agent"
                result.fallback_used = True
                return result
        
        result.error = f"All {result.attempts} attempts failed"
        return result
    
    def _is_goal_already_satisfied(self, goal: GoalStep, start_url: str) -> bool:
        """
        Check if the current state already satisfies this goal.
        
        This is the KEY to avoiding the "click listing when already on detail page" issue.
        We ask: "Is the OUTCOME we want already achieved?"
        
        Generic approach - works by checking goal type and criteria:
        1. For SELECT goals expecting navigation → check if we're already on target page
        2. For EXTRACT goals → check if we already have data
        3. For NAVIGATE goals → check if URL already matches target
        """
        criteria = goal.success_criteria
        
        # Skip check for goals that are always stateful (write, shortcut, save)
        if goal.goal_type in (GoalType.WRITE, GoalType.SHORTCUT, GoalType.SAVE):
            return False
        
        # =========================================================================
        # EXTRACT: Already have data? Skip.
        # =========================================================================
        if goal.goal_type == GoalType.EXTRACT:
            if self._extracted_data and len(self._extracted_data) >= criteria.min_extracted_count:
                self.logger.debug(f"  Already have {len(self._extracted_data)} extracted fields")
                return True
        
        # =========================================================================
        # NAVIGATE/SELECT with URL check: Already on target?
        # =========================================================================
        if goal.platform == "browser" and self.page:
            current_url = self.page.url
            
            # If goal expects us to be on a specific URL/domain
            if criteria.url_contains:
                url_check = criteria.url_contains
                # Handle template placeholders
                if "{{" in url_check and "}}" in url_check:
                    for key, value in goal.parameters.items():
                        url_check = url_check.replace(f"{{{{{key}}}}}", str(value))
                url_check = url_check.strip()
                
                if url_check and url_check.lower() in current_url.lower():
                    self.logger.debug(f"  Already on URL containing '{url_check}'")
                    return True
            
            # For SELECT goals that expect navigation to detail/listing pages
            if goal.goal_type == GoalType.SELECT and criteria.url_changed:
                # Use Gemini to check page type if we're expecting a detail page
                if "listing" in goal.goal_description.lower() or "detail" in goal.goal_description.lower():
                    page_type = self._detect_current_page_type()
                    if page_type == "detail_page":
                        self.logger.debug(f"  Already on detail page (detected via Gemini)")
                        return True
        
        # =========================================================================
        # LAUNCH: App already active?
        # =========================================================================
        if goal.goal_type == GoalType.LAUNCH:
            if criteria.app_active and self.app_launcher:
                if self.app_launcher.is_active(criteria.app_active):
                    return True
        
        return False
    
    def _detect_current_page_type(self) -> Optional[str]:
        """
        Use Gemini to detect the type of page we're currently on.
        
        Returns: 'list_page', 'detail_page', 'search_results', 'home', etc.
        """
        if not self.page or not gemini_client.is_available:
            return None
        
        try:
            screenshot = self.page.screenshot(type="png")
            
            # Ask Gemini to classify the page type
            result = gemini_client.classify_page_type(screenshot)
            if result:
                # Check both the page_type and the is_detail_page flag
                if result.get("is_detail_page"):
                    return "detail_page"
                elif result.get("is_list_page"):
                    return "list_page"
                return result.get("page_type")
        except Exception as e:
            self.logger.debug(f"Page type detection failed: {e}")
        
        return None
    
    def _handle_platform_switch(self, goal: GoalStep):
        """Handle switching between browser and desktop."""
        if goal.platform != self._current_platform:
            self.logger.debug(f"  Platform switch: {self._current_platform} → {goal.platform}")
            self._current_platform = goal.platform
        
        if goal.platform == "desktop" and goal.app_name != self._current_app:
            if self.app_launcher:
                self.logger.info(f"  Activating {goal.app_name}...")
                self.app_launcher.ensure_active(goal.app_name)
                time.sleep(1.0)  # Wait for app to become active and ready
            self._current_app = goal.app_name
    
    def _execute_strategy(self, goal: GoalStep, strategy: Strategy):
        """Execute a single strategy."""
        name = strategy.name
        
        # === SAFETY CHECKS ===
        # Check shortcuts before execution
        if strategy.shortcut_keys:
            keys = tuple(k.strip() for k in strategy.shortcut_keys.split('+'))
            check = safety_guard.check_shortcut(keys)
            if not check.allowed:
                self.logger.error(f"🛑 BLOCKED: {check.reason}")
                raise Exception(f"Safety blocked: {check.reason}")
        
        # Check typed text in terminal apps
        if strategy.input_value and goal.platform == "desktop":
            check = safety_guard.check_typed_text(strategy.input_value, goal.app_name)
            if not check.allowed:
                self.logger.error(f"🛑 BLOCKED: {check.reason}")
                raise Exception(f"Safety blocked: {check.reason}")
        
        # === BROWSER STRATEGIES ===
        if goal.platform == "browser" and self.page:
            
            if name == "selector_click" or name == "selector":
                self.page.locator(strategy.selector).first.click(timeout=5000)
            
            elif name == "text_click" or name == "text":
                self.page.get_by_text(strategy.text_match, exact=False).first.click(timeout=5000)
            
            elif name == "role_click" or name == "role":
                self.page.get_by_role(strategy.role).first.click(timeout=5000)
            
            elif name == "coordinates":
                self.page.mouse.click(strategy.coordinates[0], strategy.coordinates[1])
            
            elif name.startswith("gemini"):
                self._execute_gemini_strategy(goal, strategy)
            
            elif name in ("google_search", "search_input"):
                self._execute_search_strategy(goal, strategy)
            
            elif name == "scroll_down":
                self.page.mouse.wheel(0, 400)
                time.sleep(0.5)
            
            elif name == "scroll_up":
                self.page.mouse.wheel(0, -400)
                time.sleep(0.5)
            
            elif name == "scroll_to_content":
                # Scroll until we find content or max scrolls
                self._scroll_to_find_content(goal)
            
            elif name == "selector_type":
                elem = self.page.locator(strategy.selector).first
                elem.click()
                time.sleep(0.2)
                elem.fill(strategy.input_value or goal.parameters.get("text", ""))
            
            elif name == "focused_type":
                text = strategy.input_value or goal.parameters.get("text", "")
                self._human_type(text)
        
        # === DESKTOP STRATEGIES ===
        elif goal.platform == "desktop":
            
            if name == "activate_app":
                if self.app_launcher:
                    self.app_launcher.ensure_active(goal.app_name)
            
            elif name == "launch_app":
                if self.app_launcher:
                    self.app_launcher.launch(goal.app_name)
            
            elif name == "save_shortcut":
                import pyautogui
                pyautogui.hotkey('command', 's')
            
            elif name == "focused_type":
                text = strategy.input_value or goal.parameters.get("text", "")
                text = goal.parameters.get("filled_template", text)
                self._desktop_type(text)
            
            elif name == "coordinates":
                import pyautogui
                pyautogui.click(strategy.coordinates[0], strategy.coordinates[1])
            
            elif name.startswith("gemini"):
                self._execute_gemini_desktop_strategy(goal, strategy)
            
            elif name == "paste_content":
                self._execute_paste(goal)
    
    def _execute_search_strategy(self, goal: GoalStep, strategy: Strategy):
        """Execute a search strategy."""
        # Get the query - could be a template that needs substitution
        query = strategy.input_value or goal.template or goal.parameters.get("query", "")
        
        # Check if query still has unsubstituted placeholders
        if "{{" in query and "}}" in query:
            self.logger.warning(f"  Query has unsubstituted placeholders: {query}")
            # Try to substitute from parameters
            for key, value in goal.parameters.items():
                query = query.replace(f"{{{{{key}}}}}", str(value))
        
        self.logger.info(f"  Searching for: {query}")
        
        try:
            elem = self.page.locator(strategy.selector).first
            if elem.is_visible(timeout=2000):
                elem.click()
                time.sleep(0.2)
                elem.fill("")
                self._human_type(query)
                time.sleep(0.3)
                self.page.keyboard.press("Enter")
                self._wait_for_navigation()
                return
        except:
            pass
        
        raise Exception(f"Search strategy {strategy.name} failed")
    
    def _execute_gemini_strategy(self, goal: GoalStep, strategy: Strategy):
        """Execute a Gemini vision strategy in browser."""
        if not gemini_client.is_available:
            raise Exception("Gemini not available")
        
        screenshot = self.page.screenshot(type="png")
        description = strategy.visual_description or goal.goal_description
        
        # For extraction goals
        if goal.goal_type == GoalType.EXTRACT:
            schema = goal.extraction_schema or {}
            
            # Check if schema is specific or generic
            has_specific_schema = bool(schema) and not self._is_generic_schema(schema)
            
            if has_specific_schema:
                extracted = gemini_client.extract_fields(
                    screenshot_bytes=screenshot,
                    extraction_schema=schema
                )
            else:
                # Use auto-extraction when no specific schema
                context = goal.parameters.get("query", "") or goal.parameters.get("search_context", "")
                extracted = gemini_client.extract_page_data(
                    screenshot_bytes=screenshot,
                    context=context
                )
            
            if extracted and len(extracted) > 0:
                self._last_extracted = extracted
                return
            
            # Try scrolling to find content
            self.logger.info("  Initial extraction empty, trying with scroll...")
            if self._scroll_to_find_content(goal):
                return
            
            raise Exception("Extraction returned no data even after scrolling")
        
        # For click on listing (may need scroll)
        if strategy.name == "gemini_click_listing":
            if self._scroll_and_click_listing(description):
                return
            raise Exception(f"Could not find listing: {description}")
        
        # For click/navigation goals
        coords = gemini_client.find_element(
            screenshot_bytes=screenshot,
            element_description=description,
            screen_width=self._screen_width,
            screen_height=self._screen_height
        )
        
        if coords:
            self.logger.info(f"  Gemini found element at ({coords[0]}, {coords[1]})")
            self.page.mouse.click(coords[0], coords[1])
            self._wait_for_navigation()
        else:
            raise Exception("Gemini could not find element")
    
    def _execute_gemini_desktop_strategy(self, goal: GoalStep, strategy: Strategy):
        """Execute a Gemini vision strategy on desktop."""
        if not gemini_client.is_available:
            raise Exception("Gemini not available")
        
        import pyautogui
        import io
        
        # Take screenshot
        screenshot_pil = pyautogui.screenshot()
        img_bytes_io = io.BytesIO()
        screenshot_pil.save(img_bytes_io, format='PNG')
        screenshot_bytes = img_bytes_io.getvalue()
        
        screen_width, screen_height = screenshot_pil.size
        
        description = strategy.visual_description or goal.goal_description
        
        coords = gemini_client.find_element(
            screenshot_bytes=screenshot_bytes,
            element_description=description,
            screen_width=screen_width,
            screen_height=screen_height
        )
        
        if coords:
            self.logger.info(f"  Gemini found element at ({coords[0]}, {coords[1]})")
            pyautogui.click(coords[0], coords[1])
            time.sleep(0.3)
            
            # If strategy has input, type it
            if strategy.input_value:
                self._desktop_type(strategy.input_value)
        else:
            raise Exception("Gemini could not find element")
    
    def _execute_paste(self, goal: GoalStep):
        """Execute a paste operation with extracted data."""
        import subprocess
        import pyautogui
        
        # Get extracted data - check both sources (prefer non-empty)
        extracted = {}
        if self._extracted_data:
            extracted = self._extracted_data
        elif self._last_extracted:
            extracted = self._last_extracted
        
        self.logger.info(f"  Available data: {len(extracted)} fields")
        
        if not extracted:
            self.logger.warning("  No extracted data available!")
            return
        
        # Format extracted data as text
        lines = []
        for field, value in extracted.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."
            lines.append(f"{field}: {str_value}")
        content = "\n".join(lines)
        
        self.logger.info(f"  Pasting: {content[:80]}...")
        
        # Step 1: Ensure the target app is active and frontmost
        target_app = goal.app_name
        if self.app_launcher and target_app:
            self.logger.info(f"  Ensuring {target_app} is active...")
            self.app_launcher.ensure_active(target_app)
            time.sleep(0.5)
        
        # Step 2: Use pbcopy on macOS (more reliable than pyperclip)
        try:
            process = subprocess.Popen(
                ['pbcopy'],
                stdin=subprocess.PIPE,
                env={'LANG': 'en_US.UTF-8'}
            )
            process.communicate(content.encode('utf-8'))
            self.logger.info("  Clipboard set via pbcopy")
        except Exception as e:
            self.logger.warning(f"  pbcopy failed: {e}, trying pyperclip...")
            import pyperclip
            pyperclip.copy(content)
        
        time.sleep(0.3)
        
        # Step 3: Verify clipboard content
        try:
            result = subprocess.run(['pbpaste'], capture_output=True, text=True, timeout=2)
            if result.stdout.strip():
                self.logger.info(f"  Clipboard verified: {result.stdout[:50]}...")
            else:
                self.logger.warning("  Clipboard appears empty!")
        except:
            pass
        
        # Step 4: Click in the current window to ensure focus (for Notes)
        if target_app == "Notes":
            self.logger.info("  Clicking in Notes to focus the text area...")
            # Click in the center-right of screen (typical Notes text area)
            screen_width, screen_height = pyautogui.size()
            # Click in the main text area of Notes (roughly center-right)
            click_x = int(screen_width * 0.6)
            click_y = int(screen_height * 0.5)
            pyautogui.click(click_x, click_y)
            time.sleep(0.3)
        
        # Step 5: Use AppleScript to paste (more reliable on macOS)
        try:
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to keystroke "v" using command down'
            ], check=True, timeout=5)
            self.logger.info("  Paste command sent via AppleScript")
        except Exception as e:
            self.logger.warning(f"  AppleScript paste failed: {e}, trying pyautogui...")
            pyautogui.hotkey('command', 'v')
        
        time.sleep(0.5)
    
    def _check_success_criteria(
        self, 
        criteria: SuccessCriteria, 
        goal: GoalStep,
        start_url: str
    ) -> bool:
        """Check if success criteria are met."""
        
        # Empty criteria = always success (timeout-based)
        if criteria.is_empty() or criteria.timeout_success:
            return True
        
        # URL-based checks (browser only)
        if goal.platform == "browser" and self.page:
            current_url = self.page.url
            
            if criteria.url_changed:
                if current_url == start_url:
                    return False
            
            if criteria.url_contains:
                # Handle dynamic url_contains from site_filter parameter
                url_check = criteria.url_contains
                
                # Check if it's a template placeholder
                if "{{" in url_check and "}}" in url_check:
                    # Try to substitute from goal parameters
                    for key, value in goal.parameters.items():
                        url_check = url_check.replace(f"{{{{{key}}}}}", str(value))
                
                # Check if url_check is empty or just whitespace after substitution
                url_check = url_check.strip()
                
                # If site_filter was empty/unsubstituted, skip URL check (accept any navigation)
                if not url_check or url_check == "{{site_filter}}" or url_check == "":
                    # No specific URL required - any URL change is acceptable
                    self.logger.debug("  URL check skipped (empty site_filter)")
                else:
                    if url_check.lower() not in current_url.lower():
                        return False
            
            if criteria.url_pattern:
                import re
                if not re.search(criteria.url_pattern, current_url):
                    return False
        
        # Page type validation (using Gemini)
        if criteria.page_type and goal.platform == "browser" and self.page:
            screenshot = self.page.screenshot(type="png")
            if not gemini_client.validate_page_type(screenshot, criteria.page_type):
                self.logger.debug(f"  Page type mismatch (expected {criteria.page_type})")
                return False
        
        # Text presence check
        if criteria.page_contains_text and goal.platform == "browser" and self.page:
            try:
                page_text = self.page.inner_text("body")
                if criteria.page_contains_text not in page_text:
                    return False
            except:
                pass
        
        # App active check (desktop)
        if criteria.app_active:
            if self.app_launcher and not self.app_launcher.is_active(criteria.app_active):
                return False
        
        # Extraction criteria
        if criteria.min_extracted_count > 0:
            extracted = getattr(self, '_last_extracted', {})
            if len(extracted) < criteria.min_extracted_count:
                return False
        
        return True
    
    def _execute_agent_fallback(self, goal: GoalStep, start_url: str) -> bool:
        """Use Gemini computer-use agent as ultimate fallback."""
        if not goal.agent_goal_prompt:
            return False
        
        self.logger.info(f"  Agent goal: {goal.agent_goal_prompt}")
        
        max_agent_steps = 5
        
        for step in range(max_agent_steps):
            screenshot = self.page.screenshot(type="png") if self.page else None
            if not screenshot:
                return False
            
            # Ask Gemini what action to take
            action = gemini_client.execute_computer_use_action(
                screenshot_bytes=screenshot,
                goal=goal.agent_goal_prompt,
                screen_width=self._screen_width,
                screen_height=self._screen_height
            )
            
            if not action:
                # No more actions needed
                break
            
            # Execute the action
            self._execute_agent_action(action, goal.platform)
            time.sleep(0.5)
            
            # Check if goal achieved
            if self._check_success_criteria(goal.success_criteria, goal, start_url):
                return True
        
        # Final check
        return self._check_success_criteria(goal.success_criteria, goal, start_url)
    
    def _execute_agent_action(self, action: Dict[str, Any], platform: str):
        """Execute an action from Gemini agent."""
        name = action.get("name", "")
        args = action.get("args", {})
        
        if platform == "browser" and self.page:
            if "click" in name.lower():
                x, y = args.get("x", 0), args.get("y", 0)
                self.page.mouse.click(x, y)
            elif "type" in name.lower() or "key" in name.lower():
                text = args.get("text", "")
                self.page.keyboard.type(text)
            elif "scroll" in name.lower():
                delta = args.get("delta", 0)
                direction = args.get("direction", "down")
                # Positive delta = scroll down, negative = scroll up
                if direction == "up":
                    delta = -abs(delta) if delta else -300
                else:
                    delta = abs(delta) if delta else 300
                self.page.mouse.wheel(0, delta)
        else:
            import pyautogui
            if "click" in name.lower():
                x, y = args.get("x", 0), args.get("y", 0)
                pyautogui.click(x, y)
            elif "type" in name.lower():
                text = args.get("text", "")
                pyautogui.write(text)
            elif "scroll" in name.lower():
                delta = args.get("delta", 3)
                direction = args.get("direction", "down")
                clicks = -delta if direction == "up" else delta
                pyautogui.scroll(clicks)
    
    def _human_type(self, text: str, min_delay: float = 0.03, max_delay: float = 0.1):
        """Type with human-like delays (browser)."""
        if not self.page:
            return
        for char in text:
            self.page.keyboard.type(char)
            time.sleep(random.uniform(min_delay, max_delay))
    
    def _desktop_type(self, text: str):
        """Type on desktop."""
        import pyautogui
        import subprocess
        
        # For long text or unicode, use clipboard
        if len(text) > 50 or not all(ord(c) < 128 for c in text):
            subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
            pyautogui.hotkey('command', 'v')
        else:
            for char in text:
                if char == '\n':
                    pyautogui.press('return')
                else:
                    pyautogui.write(char, interval=0.02)
    
    def _wait_for_navigation(self, timeout: float = 5.0):
        """Wait for page to settle after navigation."""
        if not self.page:
            return
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=timeout * 1000)
        except:
            pass
        time.sleep(0.5)
    
    def _is_generic_schema(self, schema: Dict[str, Any]) -> bool:
        """Check if extraction schema is generic (should use auto-extraction)."""
        if not schema:
            return True
        
        # If schema has very specific field names, it's not generic
        specific_patterns = ["the_bier", "pizza_time", "specific_restaurant"]
        for field_name in schema.keys():
            for pattern in specific_patterns:
                if pattern in field_name.lower():
                    return False
        
        # If schema has common generic fields, it's generic
        generic_fields = {"name", "title", "address", "rating", "price", "phone", "hours"}
        schema_fields = set(f.lower() for f in schema.keys())
        if schema_fields & generic_fields:
            return True
        
        return False
    
    def _scroll_to_find_content(self, goal: GoalStep, max_scrolls: int = 5):
        """Scroll the page to find content for extraction."""
        if not self.page:
            return False
        
        schema = goal.extraction_schema or {}
        has_specific_schema = bool(schema) and not self._is_generic_schema(schema)
        context = goal.parameters.get("query", "") or goal.parameters.get("search_context", "")
        
        for i in range(max_scrolls):
            self.logger.debug(f"  Scroll attempt {i+1}/{max_scrolls}")
            
            # Scroll down
            self.page.mouse.wheel(0, 400)
            time.sleep(0.8)
            
            # Try extraction after each scroll
            screenshot = self.page.screenshot(type="png")
            
            if has_specific_schema:
                extracted = gemini_client.extract_fields(
                    screenshot_bytes=screenshot,
                    extraction_schema=schema
                )
            else:
                extracted = gemini_client.extract_page_data(
                    screenshot_bytes=screenshot,
                    context=context
                )
            
            if extracted and len(extracted) > 0:
                self._last_extracted = extracted
                self.logger.info(f"  Found content after {i+1} scrolls")
                return True
        
        return False
    
    def _scroll_and_click_listing(
        self, 
        listing_description: str, 
        max_scrolls: int = 5
    ) -> bool:
        """Scroll to find and click a listing matching the description."""
        if not self.page or not gemini_client.is_available:
            return False
        
        start_url = self.page.url
        
        for i in range(max_scrolls):
            try:
                screenshot = self.page.screenshot(type="png")
                
                coords = gemini_client.find_element(
                    screenshot_bytes=screenshot,
                    element_description=listing_description,
                    screen_width=self._screen_width,
                    screen_height=self._screen_height
                )
                
                if coords:
                    self.logger.info(f"  Found listing at ({coords[0]}, {coords[1]})")
                    self.page.mouse.click(coords[0], coords[1])
                    self._wait_for_navigation()
                    
                    # Check if we actually navigated
                    if self.page.url != start_url:
                        self.logger.info(f"  Navigated to: {self.page.url[:60]}...")
                        return True
                    else:
                        self.logger.debug(f"  Click didn't navigate, trying scroll...")
                
            except Exception as e:
                self.logger.debug(f"  Error in listing click: {e}")
            
            # Scroll down to reveal more content
            self.logger.debug(f"  Scroll {i+1}/{max_scrolls} to find listing")
            self.page.mouse.wheel(0, 400)
            time.sleep(0.8)
        
        return False

