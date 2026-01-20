"""Input capture with semantic boundaries for macOS."""
from Quartz import (
    CGEventTapCreate, CGEventTapEnable,
    kCGSessionEventTap, kCGHeadInsertEventTap,
    CGEventMaskBit, CGEventGetLocation,
    CGEventGetIntegerValueField, CGEventGetFlags,
    kCGEventLeftMouseDown, kCGEventRightMouseDown,
    kCGEventKeyDown, kCGEventScrollWheel,
    kCGKeyboardEventKeycode,
    kCGEventFlagMaskCommand, kCGEventFlagMaskShift,
    kCGEventFlagMaskControl, kCGEventFlagMaskAlternate,
    CFRunLoopAddSource, CFRunLoopGetCurrent,
    kCFRunLoopDefaultMode, CFRunLoopRun, CFRunLoopStop,
    CFMachPortCreateRunLoopSource,
)
from threading import Thread, Lock
import time
from typing import Callable, Optional, List
from src.models.session_artifact import InputEvent
from src.utils.logger import setup_logger
from src.utils.config import config
import pyperclip


# Key code to character mapping (macOS virtual key codes)
KEY_MAP = {
    0: 'a', 1: 's', 2: 'd', 3: 'f', 4: 'h', 5: 'g', 6: 'z', 7: 'x',
    8: 'c', 9: 'v', 11: 'b', 12: 'q', 13: 'w', 14: 'e', 15: 'r',
    16: 'y', 17: 't', 31: 'o', 32: 'u', 34: 'i', 35: 'p',
    37: 'l', 38: 'j', 40: 'k', 41: ';', 45: 'n', 46: 'm',
    43: ',', 47: '.', 44: '/', 
    18: '1', 19: '2', 20: '3', 21: '4', 23: '5',
    22: '6', 26: '7', 28: '8', 25: '9', 29: '0',
    27: '-', 24: '=', 33: '[', 30: ']', 42: '\\', 39: "'",
    50: '`',
    # Special keys
    36: 'return', 48: 'tab', 49: 'space', 51: 'delete', 53: 'escape',
    123: 'left', 124: 'right', 125: 'down', 126: 'up',
}

# Modifier key shortcuts
MODIFIER_SHORTCUTS = {
    ('cmd', 'c'): 'copy',
    ('cmd', 'v'): 'paste',
    ('cmd', 's'): 'save',
    ('cmd', 'a'): 'select_all',
    ('cmd', 'z'): 'undo',
    ('cmd', 'shift', 'z'): 'redo',
    ('cmd', 'f'): 'find',
    ('cmd', 'tab'): 'app_switch',
    ('cmd', 'w'): 'close_tab',
    ('cmd', 't'): 'new_tab',
    ('cmd', 'n'): 'new_window',
    ('cmd', 'q'): 'quit_app',
}


class InputCapture:
    """
    Captures keyboard and mouse input with semantic buffering.
    
    Key principles:
    - Buffer typing until semantic boundary (not character count)
    - Detect modifier key shortcuts
    - Flush on: click, submit, app switch, idle+action, shortcuts
    - IMPROVED: Better clipboard capture timing for copy events
    """
    
    def __init__(self, callback: Callable[[InputEvent], None]):
        """
        Initialize input capture.
        
        Args:
            callback: Function to call when input event is ready
        """
        self.callback = callback
        self.logger = setup_logger("InputCapture")
        
        # Timing
        self.start_time = 0.0
        self.is_running = False
        
        # Typing buffer
        self._typing_buffer: List[str] = []
        self._last_keystroke_time = 0.0
        self._typing_lock = Lock()
        
        # Event tap
        self.tap = None
        self.thread = None
        self.run_loop_source = None
        
        # Idle detection
        self._idle_check_thread = None
        
        # =====================================================================
        # CLIPBOARD STATE TRACKING (FIX for stale clipboard bug)
        # =====================================================================
        self._previous_clipboard: Optional[str] = None
        self._clipboard_lock = Lock()
    
    def _get_timestamp(self) -> float:
        """Get current timestamp relative to start."""
        return time.time() - self.start_time
    
    def _flush_typing_buffer(self, reason: str):
        """Flush accumulated typing as a single event."""
        with self._typing_lock:
            if self._typing_buffer:
                text = ''.join(self._typing_buffer)
                timestamp = self._get_timestamp()
                
                event = InputEvent(
                    type="keyboard",
                    timestamp=timestamp,
                    text=text,
                    flush_reason=reason
                )
                self.callback(event)
                
                self.logger.debug(f"Flushed typing: '{text[:30]}...' (reason: {reason})")
                self._typing_buffer.clear()
    
    def _get_modifiers(self, flags: int) -> List[str]:
        """Extract active modifier keys from event flags."""
        modifiers = []
        if flags & kCGEventFlagMaskCommand:
            modifiers.append('cmd')
        if flags & kCGEventFlagMaskShift:
            modifiers.append('shift')
        if flags & kCGEventFlagMaskControl:
            modifiers.append('ctrl')
        if flags & kCGEventFlagMaskAlternate:
            modifiers.append('alt')
        return modifiers
    
    def _check_shortcut(self, modifiers: List[str], key: str) -> Optional[str]:
        """Check if modifier+key combination is a known shortcut."""
        # Try exact match
        combo = tuple(sorted(modifiers) + [key])
        
        # Check all possible combinations
        for shortcut_combo, shortcut_name in MODIFIER_SHORTCUTS.items():
            if set(shortcut_combo) == set(combo):
                return shortcut_name
        
        # Check without sorting (for order-sensitive)
        for shortcut_combo, shortcut_name in MODIFIER_SHORTCUTS.items():
            combo_set = set(modifiers + [key])
            shortcut_set = set(shortcut_combo)
            if combo_set == shortcut_set:
                return shortcut_name
        
        return None
    
    def _capture_clipboard_with_retry(self, max_retries: int = 3) -> Optional[str]:
        """
        Capture clipboard content with retry logic.
        
        FIX: The previous 0.05s delay was too short. The OS needs time to
        complete the copy operation before the clipboard is updated.
        
        This method:
        1. Waits longer (0.2s initial)
        2. Retries up to max_retries times
        3. Compares against previous clipboard to detect actual changes
        """
        with self._clipboard_lock:
            # Initial delay - let OS complete the copy operation
            time.sleep(0.2)
            
            for attempt in range(max_retries):
                try:
                    new_clipboard = pyperclip.paste()
                    
                    # Check if clipboard actually changed (not stale data)
                    if new_clipboard and new_clipboard != self._previous_clipboard:
                        self._previous_clipboard = new_clipboard
                        self.logger.debug(
                            f"Captured clipboard (attempt {attempt + 1}): "
                            f"'{new_clipboard[:50]}...'" if len(new_clipboard) > 50 
                            else f"'{new_clipboard}'"
                        )
                        return new_clipboard
                    
                    # If same as previous, wait a bit more and retry
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        
                except Exception as e:
                    self.logger.debug(f"Clipboard read attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
            
            # If we couldn't get new content, return whatever we have
            # This is better than returning None when the user did copy something
            try:
                current = pyperclip.paste()
                if current:
                    self.logger.warning(
                        f"Could not detect clipboard change, using current content: "
                        f"'{current[:30]}...'"
                    )
                    self._previous_clipboard = current
                    return current
            except:
                pass
            
            self.logger.warning("Failed to capture clipboard content after all retries")
            return None
    
    def _event_handler(self, proxy, event_type, event, refcon):
        """Handle captured events."""
        if not self.is_running:
            return event
        
        try:
            timestamp = self._get_timestamp()
            
            # Check for idle timeout (flush if typing paused)
            with self._typing_lock:
                if self._typing_buffer and self._last_keystroke_time > 0:
                    idle_time = timestamp - self._last_keystroke_time
                    if idle_time > config.typing_idle_threshold:
                        # Will be flushed by next action
                        pass
            
            # Mouse click events
            if event_type in (kCGEventLeftMouseDown, kCGEventRightMouseDown):
                location = CGEventGetLocation(event)
                
                # Flush any pending typing BEFORE the click
                self._flush_typing_buffer("click")
                
                button = "left" if event_type == kCGEventLeftMouseDown else "right"
                
                input_event = InputEvent(
                    type="mouse_click",
                    timestamp=timestamp,
                    x=int(location.x),
                    y=int(location.y),
                    button=button
                )
                self.callback(input_event)
            
            # Keyboard events
            elif event_type == kCGEventKeyDown:
                keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                flags = CGEventGetFlags(event)
                modifiers = self._get_modifiers(flags)
                
                key_char = KEY_MAP.get(keycode, f"key_{keycode}")
                
                # Check for shortcuts first
                if modifiers:
                    shortcut = self._check_shortcut(modifiers, key_char)
                    if shortcut:
                        # Flush typing before shortcut
                        self._flush_typing_buffer("shortcut")
                        
                        # Capture clipboard content on copy - WITH IMPROVED TIMING
                        clipboard_content = None
                        if shortcut == "copy":
                            clipboard_content = self._capture_clipboard_with_retry()
                        
                        input_event = InputEvent(
                            type="keyboard_shortcut",
                            timestamp=timestamp,
                            shortcut=shortcut,
                            clipboard_content=clipboard_content
                        )
                        self.callback(input_event)
                        return event
                
                # Handle special keys
                if key_char == 'return':
                    # Flush typing WITH the return (it's part of the action)
                    self._flush_typing_buffer("submit")
                    
                    input_event = InputEvent(
                        type="keyboard",
                        timestamp=timestamp,
                        key="return"
                    )
                    self.callback(input_event)
                
                elif key_char == 'tab':
                    self._flush_typing_buffer("tab")
                    
                    input_event = InputEvent(
                        type="keyboard",
                        timestamp=timestamp,
                        key="tab"
                    )
                    self.callback(input_event)
                
                elif key_char == 'escape':
                    self._flush_typing_buffer("escape")
                    
                    input_event = InputEvent(
                        type="keyboard",
                        timestamp=timestamp,
                        key="escape"
                    )
                    self.callback(input_event)
                
                elif key_char == 'space':
                    # Space is part of typing
                    with self._typing_lock:
                        self._typing_buffer.append(' ')
                        self._last_keystroke_time = timestamp
                
                elif key_char == 'delete':
                    # Handle backspace - remove last char from buffer
                    with self._typing_lock:
                        if self._typing_buffer:
                            self._typing_buffer.pop()
                        self._last_keystroke_time = timestamp
                
                elif len(key_char) == 1:
                    # Regular printable character
                    with self._typing_lock:
                        # Handle shift for uppercase AND special chars
                        if 'shift' in modifiers:
                            if key_char.isalpha():
                                key_char = key_char.upper()
                            else:
                                # Shift + number = special character
                                shift_map = {
                                    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
                                    '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
                                    '-': '_', '=': '+', '[': '{', ']': '}', '\\': '|',
                                    ';': ':', "'": '"', ',': '<', '.': '>', '/': '?',
                                    '`': '~'
                                }
                                key_char = shift_map.get(key_char, key_char)
                        
                        self._typing_buffer.append(key_char)
                        self._last_keystroke_time = timestamp
            
            # Scroll events (don't flush typing, just log)
            elif event_type == kCGEventScrollWheel:
                # We track scroll but don't create events for every scroll
                pass
        
        except Exception as e:
            self.logger.error(f"Error in event handler: {e}")
        
        return event
    
    def _idle_checker(self):
        """Background thread to check for typing idle timeout."""
        while self.is_running:
            time.sleep(0.1)  # Check every 100ms
            
            with self._typing_lock:
                if self._typing_buffer and self._last_keystroke_time > 0:
                    idle_time = self._get_timestamp() - self._last_keystroke_time
                    # Don't auto-flush on idle alone
                    # Only flush when combined with next action
    
    def start(self):
        """Start capturing input events."""
        self.logger.info("Starting input capture...")
        self.is_running = True
        self.start_time = time.time()
        
        # Initialize clipboard state
        try:
            self._previous_clipboard = pyperclip.paste()
            self.logger.debug(f"Initial clipboard: '{self._previous_clipboard[:30] if self._previous_clipboard else 'empty'}...'")
        except:
            self._previous_clipboard = None
        
        # Create event tap
        event_mask = (
            CGEventMaskBit(kCGEventLeftMouseDown) |
            CGEventMaskBit(kCGEventRightMouseDown) |
            CGEventMaskBit(kCGEventKeyDown) |
            CGEventMaskBit(kCGEventScrollWheel)
        )
        
        self.tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            0,  # Active filter (not passive)
            event_mask,
            self._event_handler,
            None
        )
        
        if not self.tap:
            raise RuntimeError(
                "Failed to create event tap. "
                "Grant Accessibility permissions in System Preferences > Security & Privacy > Accessibility"
            )
        
        # Start event tap in separate thread
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        # Start idle checker
        self._idle_check_thread = Thread(target=self._idle_checker, daemon=True)
        self._idle_check_thread.start()
        
        self.logger.info("Input capture started")
    
    def _run_loop(self):
        """Run the event tap loop."""
        self.run_loop_source = CFMachPortCreateRunLoopSource(None, self.tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), self.run_loop_source, kCFRunLoopDefaultMode)
        CGEventTapEnable(self.tap, True)
        CFRunLoopRun()
    
    def stop(self):
        """Stop capturing events."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping input capture...")
        self.is_running = False
        
        # Flush any remaining typing
        self._flush_typing_buffer("stop")
        
        # Stop the run loop
        if self.tap:
            CGEventTapEnable(self.tap, False)
            CFRunLoopStop(CFRunLoopGetCurrent())
        
        self.logger.info("Input capture stopped")
    
    def force_flush(self, reason: str = "forced"):
        """Force flush the typing buffer (called externally on app switch, etc.)."""
        self._flush_typing_buffer(reason)