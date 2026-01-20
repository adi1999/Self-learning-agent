"""Event-driven screen capture for macOS."""
import mss
import mss.tools
from pathlib import Path
from typing import Optional, Set
import time
import hashlib
from src.utils.logger import setup_logger
from src.utils.config import config


class ScreenCapture:
    """
    Event-driven screen capture.
    
    Takes screenshots on specific triggers rather than at fixed intervals.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize screen capture.
        
        Args:
            output_dir: Directory to save screenshots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("ScreenCapture")
        
        self.sct = mss.mss()
        self._last_capture_time = 0.0
        self._last_screenshot_hash: Optional[str] = None
        self._capture_count = 0
        
        # Track which triggers have been processed
        self._processed_triggers: Set[str] = set()
    
    def should_capture(self, trigger: str, timestamp: float) -> bool:
        """
        Determine if we should take a screenshot for this trigger.
        
        Args:
            trigger: Type of trigger (click, submit, page_load, etc.)
            timestamp: Current timestamp
        
        Returns:
            True if should capture
        """
        # Always capture for configured triggers
        if trigger in config.screenshot_triggers:
            return True
        
        # Backup capture if too long since last screenshot
        if timestamp - self._last_capture_time > config.backup_screenshot_interval:
            return True
        
        return False
    
    def capture(self, trigger: str, timestamp: float) -> Optional[str]:
        """
        Capture a screenshot if appropriate for the trigger.
        
        Args:
            trigger: What triggered this capture
            timestamp: Current recording timestamp
        
        Returns:
            Relative path to saved screenshot, or None if skipped
        """
        if not self.should_capture(trigger, timestamp):
            return None
        
        try:
            # Capture primary monitor
            monitor = self.sct.monitors[1]
            screenshot = self.sct.grab(monitor)
            
            # Check if screenshot is different from last (avoid duplicates)
            screenshot_bytes = screenshot.rgb
            current_hash = hashlib.md5(screenshot_bytes).hexdigest()[:16]
            
            if current_hash == self._last_screenshot_hash and trigger != "page_load":
                self.logger.debug(f"Skipping duplicate screenshot for {trigger}")
                return None
            
            self._last_screenshot_hash = current_hash
            
            # Generate filename
            self._capture_count += 1
            filename = f"screen_{timestamp:.3f}_{trigger}_{self._capture_count}.png"
            filepath = self.output_dir / filename
            
            # Save screenshot
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filepath))
            
            self._last_capture_time = timestamp
            self.logger.debug(f"Captured screenshot: {filename} (trigger: {trigger})")
            
            return f"screenshots/{filename}"
        
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def capture_now(self, output_path: Path) -> Optional[Path]:
        """
        Capture a screenshot immediately to a specific path.
        
        Args:
            output_path: Where to save the screenshot
        
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            monitor = self.sct.monitors[1]
            screenshot = self.sct.grab(monitor)
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(output_path))
            return output_path
        except Exception as e:
            self.logger.error(f"Immediate capture failed: {e}")
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int, output_path: Path) -> Optional[Path]:
        """
        Capture a specific region of the screen.
        
        Args:
            x, y: Top-left corner
            width, height: Region size
            output_path: Where to save
        
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            region = {"left": x, "top": y, "width": width, "height": height}
            screenshot = self.sct.grab(region)
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(output_path))
            return output_path
        except Exception as e:
            self.logger.error(f"Region capture failed: {e}")
            return None
    
    def get_screen_size(self) -> tuple:
        """Get primary screen dimensions."""
        monitor = self.sct.monitors[1]
        return (monitor["width"], monitor["height"])
    
    def close(self):
        """Clean up resources."""
        if self.sct:
            self.sct.close()
            self.logger.debug("Screen capture closed")