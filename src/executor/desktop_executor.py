"""Desktop action execution using macapptree and pyautogui."""
import time
import io
import pyautogui
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.models.workflow_recipe import WorkflowStep, ElementReference
from src.executor.app_launcher import AppLauncher
from src.observer.desktop_capture import DesktopCapture
from src.utils.logger import setup_logger
from src.utils.config import config
from src.utils.gemini_client import gemini_client
from src.utils.safety_guard import safety_guard
import pyperclip

# Configure pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


@dataclass
class DesktopStepResult:
    """Result of desktop step execution."""
    success: bool
    error: Optional[str] = None
    strategy_used: Optional[str] = None


class DesktopExecutor:
    """
    Executes desktop steps using macapptree and pyautogui.
    
    Strategies:
    1. Accessibility-based (role + name)
    2. Text-based (find by visible text)
    3. Position-based (with tolerance)
    4. Coordinate-based (fallback)
    """
    
    def __init__(self):
        self.logger = setup_logger("DesktopExecutor")
        self.app_launcher = AppLauncher()
        self.desktop_capture = DesktopCapture()
        
        self._current_app: Optional[str] = None
    
    def ensure_app_active(self, app_name: str) -> bool:
        """Ensure the specified app is active."""
        if self.app_launcher.is_active(app_name):
            self._current_app = app_name
            return True
        
        success = self.app_launcher.ensure_active(app_name)
        
        if success:
            self._current_app = app_name
            time.sleep(0.5)  # Let app settle
        
        return success
    
    def execute_step(self, step: WorkflowStep) -> DesktopStepResult:
        """Execute a single desktop step."""
        
        # Ensure correct app is active
        if step.app_name != self._current_app:
            self.logger.info(f"Switching to {step.app_name}")
            if not self.ensure_app_active(step.app_name):
                return DesktopStepResult(
                    success=False,
                    error=f"Could not activate {step.app_name}"
                )
        
        if step.action_type == "launch_app":
            return self._execute_launch(step)
        elif step.action_type == "click":
            return self._execute_click(step)
        elif step.action_type == "type":
            return self._execute_type(step)
        elif step.action_type == "shortcut":
            return self._execute_shortcut(step)
        else:
            self.logger.warning(f"Unknown desktop action: {step.action_type}")
            return DesktopStepResult(success=True)
    
    def _execute_launch(self, step: WorkflowStep) -> DesktopStepResult:
        """Execute app launch."""
        app_name = step.parameter_bindings.get("app_name", step.app_name)
        
        # Use ensure_app_active for robust activation
        self.logger.info(f"Launching/Activating {app_name}...")
        success = self.ensure_app_active(app_name)
        
        if success:
            self._current_app = app_name
            # Wait a bit longer for window to settle and appear in screenshot
            time.sleep(2.0)
            return DesktopStepResult(success=True, strategy_used="app_launch")
        
        return DesktopStepResult(success=False, error=f"Failed to launch or activate {app_name}")
    
    def _execute_click(self, step: WorkflowStep) -> DesktopStepResult:
        """Execute click step."""
        ref = step.element_reference
        
        if not ref:
            return DesktopStepResult(success=False, error="No element reference")
        
        # Get current app tree if available
        tree = None
        if self.desktop_capture.is_available and self._current_app:
            tree = self.desktop_capture.capture_app_tree(self._current_app)
        
        # Try strategies - now includes Gemini fallback
        strategies = [
            ("accessibility", lambda: self._click_by_accessibility(ref, tree)),
            ("text", lambda: self._click_by_text(ref, tree)),
            ("position", lambda: self._click_by_position(ref, tree)),
            ("gemini_vision", lambda: self._click_by_gemini_vision(ref, step)),
            ("coordinates", lambda: self._click_by_coordinates(ref)),
        ]
        
        for name, strategy_fn in strategies:
            try:
                success = strategy_fn()
                if success:
                    time.sleep(0.3)
                    return DesktopStepResult(success=True, strategy_used=name)
            except Exception as e:
                self.logger.debug(f"Click strategy {name} failed: {e}")
                continue
        
        return DesktopStepResult(success=False, error="Could not click element")
    
    def _click_by_accessibility(self, ref: ElementReference, tree: Optional[Dict]) -> bool:
        """Click element by accessibility role and name."""
        if not tree or not ref.accessibility_role:
            return False
        
        element = self.desktop_capture.find_element_by_role_name(
            tree,
            ref.accessibility_role,
            ref.accessibility_name or ""
        )
        
        if element:
            return self._click_element(element)
        
        return False
    
    def _click_by_text(self, ref: ElementReference, tree: Optional[Dict]) -> bool:
        """Click element by visible text."""
        if not tree or not ref.text:
            return False
        
        def find_by_text(node: Dict, target: str) -> Optional[Dict]:
            name = node.get("name", "") or ""
            value = node.get("value", "") or ""
            
            if target.lower() in name.lower() or target.lower() in value.lower():
                return node
            
            for child in node.get("children", []):
                result = find_by_text(child, target)
                if result:
                    return result
            
            return None
        
        element = find_by_text(tree, ref.text)
        
        if element:
            return self._click_element(element)
        
        return False
    
    def _click_by_position(self, ref: ElementReference, tree: Optional[Dict]) -> bool:
        """Click element by approximate position."""
        if not tree or not ref.absolute_position:
            return False
        
        try:
            pos_parts = ref.absolute_position.split(";")
            target_x = float(pos_parts[0])
            target_y = float(pos_parts[1])
        except:
            return False
        
        # Find element near this position
        element = self._find_nearest_element(tree, target_x, target_y, tolerance=50)
        
        if element:
            return self._click_element(element)
        
        return False
    
    def _click_by_coordinates(self, ref: ElementReference) -> bool:
        """Click at raw coordinates (fallback)."""
        coords = ref.coordinates
        
        if not coords or len(coords) < 2:
            return False
        
        pyautogui.click(coords[0], coords[1])
        return True
    
    def _click_by_gemini_vision(self, ref: ElementReference, step: WorkflowStep) -> bool:
        """
        Use Gemini vision to find and click element.
        
        This is the key fallback for desktop apps where accessibility tree
        and text matching fail.
        """
        if not gemini_client.is_available:
            self.logger.debug("Gemini not available for vision fallback")
            return False
        
        self.logger.info("  Trying Gemini vision fallback for desktop...")
        
        # Take screenshot of the current screen
        try:
            # pyautogui.screenshot() returns a PIL Image
            screenshot_pil = pyautogui.screenshot()
            
            # Convert to bytes for Gemini
            img_bytes_io = io.BytesIO()
            screenshot_pil.save(img_bytes_io, format='PNG')
            screenshot_bytes = img_bytes_io.getvalue()
            
            # Get screen dimensions
            screen_width, screen_height = screenshot_pil.size
        except Exception as e:
            self.logger.debug(f"Screenshot capture failed: {e}")
            return False
        
        # Build description for Gemini
        description = self._build_gemini_element_description(ref, step)
        self.logger.info(f"  Gemini searching for: {description}")
        
        # Call Gemini
        try:
            coords = gemini_client.find_element(
                screenshot_bytes=screenshot_bytes,
                element_description=description,
                screen_width=screen_width,
                screen_height=screen_height
            )
            
            if coords:
                self.logger.info(f"  Gemini found element at ({coords[0]}, {coords[1]})")
                pyautogui.click(coords[0], coords[1])
                return True
            else:
                self.logger.debug("  Gemini did not find the element")
                return False
        except Exception as e:
            self.logger.debug(f"Gemini find_element failed: {e}")
            return False
    
    def _build_gemini_element_description(self, ref: ElementReference, step: WorkflowStep) -> str:
        """Build a useful description for Gemini element finding."""
        parts = []
        
        # Use visual hint if available (now contains rich desktop info)
        if ref.visual_hint:
            parts.append(ref.visual_hint)
        
        # Add text content for matching
        if ref.text:
            parts.append(f'with text "{ref.text[:40]}"')
        
        # Add accessibility info
        if ref.accessibility_role:
            parts.append(f"(role: {ref.accessibility_role})")
        
        # Add positional hint if we have bbox
        if ref.bbox and len(ref.bbox) >= 4:
            x, y, w, h = ref.bbox[:4]
            parts.append(f"approximately at position ({x}, {y})")
        
        # Intent-based fallback
        if not parts:
            intent_hints = {
                "select": "clickable element or button",
                "write": "text input area or text field",
                "navigate": "navigation link or menu item",
            }
            parts.append(intent_hints.get(step.intent, "interactive element"))
            parts.append(f"in {step.app_name}")
        
        return " ".join(parts)
    
    def _click_element(self, element: Dict) -> bool:
        """Click on an accessibility element."""
        # Get center of element
        abs_pos = element.get("absolute_position", "")
        size = element.get("size", "")
        
        if not abs_pos or not size or ";" not in abs_pos or ";" not in size:
            return False
        
        try:
            pos_parts = abs_pos.split(";")
            size_parts = size.split(";")
            
            x = float(pos_parts[0]) + float(size_parts[0]) / 2
            y = float(pos_parts[1]) + float(size_parts[1]) / 2
            
            pyautogui.click(x, y)
            return True
        except Exception as e:
            self.logger.debug(f"Click element failed: {e}")
            return False
    
    def _find_nearest_element(
        self,
        tree: Dict,
        target_x: float,
        target_y: float,
        tolerance: float
    ) -> Optional[Dict]:
        """Find element nearest to target position."""
        best_element = None
        best_distance = float('inf')
        
        def search(node: Dict):
            nonlocal best_element, best_distance
            
            abs_pos = node.get("absolute_position", "")
            size = node.get("size", "")
            
            if abs_pos and size and ";" in abs_pos and ";" in size:
                try:
                    pos_parts = abs_pos.split(";")
                    size_parts = size.split(";")
                    
                    center_x = float(pos_parts[0]) + float(size_parts[0]) / 2
                    center_y = float(pos_parts[1]) + float(size_parts[1]) / 2
                    
                    distance = ((center_x - target_x) ** 2 + (center_y - target_y) ** 2) ** 0.5
                    
                    if distance < tolerance and distance < best_distance:
                        best_distance = distance
                        best_element = node
                except:
                    pass
            
            for child in node.get("children", []):
                search(child)
        
        search(tree)
        return best_element
    
    def _execute_type(self, step: WorkflowStep) -> DesktopStepResult:
        """Execute typing step - KEY: handles template-filled content."""
        value = step.parameter_bindings.get("value", "")
        
        if not value:
            return DesktopStepResult(success=False, error="No value to type")
        
        # === SAFETY CHECK: Block dangerous commands in terminal apps ===
        check = safety_guard.check_typed_text(value, step.app_name)
        if not check.allowed:
            self.logger.error(f"🛑 BLOCKED command: {check.reason}")
            return DesktopStepResult(success=False, error=f"Blocked: {check.reason}")
        
        self.logger.info(f"  Typing: {value[:50]}...")
        
        # Click element first if specified
        if step.element_reference:
            click_result = self._execute_click(step)
            if not click_result.success:
                self.logger.warning("Could not click target, typing anyway")
            time.sleep(0.2)
        
        # Type the value
        try:
            # Use pyautogui.write for simple ASCII
            # Fall back to clipboard for Unicode or long text
            if len(value) > 100 or not all(ord(c) < 128 for c in value):
                # Use clipboard for long text or Unicode
                import subprocess
                subprocess.run(['pbcopy'], input=value.encode('utf-8'), check=True)
                pyautogui.hotkey('command', 'v')
                self.logger.debug("  Used clipboard method for typing")
            else:
                # Type character by character with small delay
                for char in value:
                    if char == '\n':
                        pyautogui.press('return')
                    elif ord(char) < 128:
                        pyautogui.write(char, interval=0.02)
                    else:
                        # Single Unicode char via clipboard
                        import subprocess
                        subprocess.run(['pbcopy'], input=char.encode('utf-8'), check=True)
                        pyautogui.hotkey('command', 'v')
            
            time.sleep(0.2)
            return DesktopStepResult(success=True, strategy_used="pyautogui")
        
        except Exception as e:
            return DesktopStepResult(success=False, error=str(e))
    
    def _execute_shortcut(self, step: WorkflowStep) -> DesktopStepResult:
        """Execute keyboard shortcut."""
        shortcut = step.shortcut
        
        # === SAFETY CHECK ===
        if shortcut:
            # Convert shortcut name to key tuple for checking
            shortcut_to_keys = {
                "save": ("command", "s"),
                "copy": ("command", "c"),
                "paste": ("command", "v"),
                "quit": ("command", "q"),
                "close": ("command", "w"),
            }
            keys = shortcut_to_keys.get(shortcut, (shortcut,))
            check = safety_guard.check_shortcut(keys)
            if not check.allowed:
                self.logger.error(f"🛑 BLOCKED shortcut: {check.reason}")
                return DesktopStepResult(success=False, error=f"Blocked: {check.reason}")
        
        # If pasting, set clipboard content first
        if shortcut == "paste":
            # Set clipboard content if provided
            if step.clipboard_content:
                pyperclip.copy(step.clipboard_content)
                self.logger.info(f"  Set clipboard before paste: {step.clipboard_content[:50]}...")
            
            # =========================================================================
            # FIX: Add delay and use more reliable key pressing
            # =========================================================================
            time.sleep(0.1)  # Let clipboard settle
            
            # Method 1: Try AppleScript for more reliable paste (macOS)
            try:
                import subprocess
                subprocess.run([
                    'osascript', '-e',
                    'tell application "System Events" to keystroke "v" using command down'
                ], check=True, timeout=2)
                time.sleep(0.1)
                return DesktopStepResult(success=True, strategy_used="applescript_paste")
            except Exception as e:
                self.logger.debug(f"AppleScript paste failed: {e}, trying pyautogui")
            
            # Method 2: Fallback to pyautogui with explicit key handling
            try:
                import pyautogui
                # Ensure no keys are stuck
                pyautogui.keyUp('command')
                pyautogui.keyUp('v')
                time.sleep(0.05)
                
                # Use press() with explicit interval
                pyautogui.hotkey('command', 'v', interval=0.05)
                time.sleep(0.1)
                return DesktopStepResult(success=True, strategy_used="pyautogui_paste")
            except Exception as e:
                return DesktopStepResult(success=False, error=f"Paste failed: {e}")
        
        shortcut_keys = {
            "save": ('command', 's'),
            "copy": ('command', 'c'),
            "paste": ('command', 'v'),
            "undo": ('command', 'z'),
            "redo": ('command', 'shift', 'z'),
            "select_all": ('command', 'a'),
            "find": ('command', 'f'),
            "new": ('command', 'n'),
            "close": ('command', 'w'),
            "quit": ('command', 'q'),
        }
        
        keys = shortcut_keys.get(shortcut)
        
        if not keys:
            # Try to parse shortcut string
            if shortcut:
                keys = tuple(shortcut.lower().split('+'))
        
        if not keys:
            return DesktopStepResult(success=False, error=f"Unknown shortcut: {shortcut}")
        
        try:
            pyautogui.hotkey(*keys)
            time.sleep(0.3)
            return DesktopStepResult(success=True, strategy_used="pyautogui_hotkey")
        
        except Exception as e:
            return DesktopStepResult(success=False, error=str(e))
    
    def press_enter(self):
        """Press Enter key."""
        pyautogui.press('return')
    
    def press_tab(self):
        """Press Tab key."""
        pyautogui.press('tab')