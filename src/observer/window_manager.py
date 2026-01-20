"""Window and application tracking for macOS."""
from AppKit import NSWorkspace
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID
)
from typing import Tuple, Optional, Dict, Any
from src.utils.logger import setup_logger


class WindowManager:
    """Tracks active application and window information on macOS."""
    
    # Known browsers
    BROWSERS = ['chrome', 'safari', 'firefox', 'edge', 'brave', 'arc', 'opera']
    
    def __init__(self):
        self.workspace = NSWorkspace.sharedWorkspace()
        self.logger = setup_logger("WindowManager")
        
        # Cache last known state
        self._last_app: Optional[str] = None
        self._last_window: Optional[str] = None
    
    def get_active_app(self) -> str:
        """Get the name of the currently active application."""
        try:
            active_app = self.workspace.activeApplication()
            if active_app:
                app_name = active_app.get('NSApplicationName', 'Unknown')
                self._last_app = app_name
                return app_name
        except Exception as e:
            self.logger.error(f"Error getting active app: {e}")
        
        return self._last_app or "Unknown"
    
    def get_active_window_title(self) -> str:
        """Get the title of the currently active window."""
        try:
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID
            )
            
            if not window_list:
                return self._last_window or "Unknown"
            
            # Find the frontmost window (layer 0)
            for window in window_list:
                layer = window.get('kCGWindowLayer', -1)
                if layer == 0:
                    title = window.get('kCGWindowName', '')
                    if title:
                        self._last_window = title
                        return title
            
            return self._last_window or "Unknown"
        
        except Exception as e:
            self.logger.error(f"Error getting window title: {e}")
            return self._last_window or "Unknown"
    
    def get_active_context(self) -> Tuple[str, str]:
        """
        Get both active app and window title.
        
        Returns:
            Tuple of (app_name, window_title)
        """
        return (self.get_active_app(), self.get_active_window_title())
    
    def is_browser(self, app_name: Optional[str] = None) -> bool:
        """
        Check if the given app (or current app) is a web browser.
        
        Args:
            app_name: App name to check, or None for current app
        
        Returns:
            True if app is a browser
        """
        if app_name is None:
            app_name = self.get_active_app()
        
        app_lower = app_name.lower()
        return any(browser in app_lower for browser in self.BROWSERS)
    
    def get_platform(self, app_name: Optional[str] = None) -> str:
        """
        Get platform type for the app.
        
        Returns:
            "browser" or "desktop"
        """
        return "browser" if self.is_browser(app_name) else "desktop"
    
    def did_app_change(self, previous_app: str) -> bool:
        """Check if app changed from previous."""
        current = self.get_active_app()
        return current != previous_app
    
    def get_window_bounds(self, app_name: Optional[str] = None) -> Optional[Dict[str, int]]:
        """
        Get the bounds of the active window.
        
        Returns:
            Dict with x, y, width, height or None
        """
        try:
            target_app = app_name or self.get_active_app()
            
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID
            )
            
            if not window_list:
                return None
            
            for window in window_list:
                owner = window.get('kCGWindowOwnerName', '')
                if target_app.lower() in owner.lower():
                    bounds = window.get('kCGWindowBounds', {})
                    if bounds:
                        return {
                            'x': int(bounds.get('X', 0)),
                            'y': int(bounds.get('Y', 0)),
                            'width': int(bounds.get('Width', 0)),
                            'height': int(bounds.get('Height', 0))
                        }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting window bounds: {e}")
            return None
    
    def get_running_apps(self) -> list:
        """Get list of running application names."""
        try:
            apps = self.workspace.runningApplications()
            return [app.localizedName() for app in apps if app.localizedName()]
        except Exception:
            return []