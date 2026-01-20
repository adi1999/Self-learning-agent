"""Desktop application launching and management for macOS."""
import subprocess
import time
from typing import Optional
from AppKit import NSWorkspace, NSRunningApplication
from src.utils.logger import setup_logger


class AppLauncher:
    """
    Launches and manages desktop applications on macOS.
    
    Handles:
    - Launching apps by name or bundle ID
    - Activating/focusing apps
    - Checking if app is running
    - Waiting for app to be ready
    """
    
    # Common app bundle IDs
    BUNDLE_IDS = {
        "chrome": "com.google.Chrome",
        "brave": "com.brave.Browser",
        "safari": "com.apple.Safari",
        "firefox": "org.mozilla.firefox",
        "notes": "com.apple.Notes",
        "textedit": "com.apple.TextEdit",
        "finder": "com.apple.finder",
        "terminal": "com.apple.Terminal",
        "vscode": "com.microsoft.VSCode",
        "code": "com.microsoft.VSCode",
        "slack": "com.tinyspeck.slackmacgap",
        "excel": "com.microsoft.Excel",
        "word": "com.microsoft.Word",
        "pages": "com.apple.iWork.Pages",
        "numbers": "com.apple.iWork.Numbers",
    }
    
    def __init__(self):
        self.workspace = NSWorkspace.sharedWorkspace()
        self.logger = setup_logger("AppLauncher")
    
    def launch(self, app_name: str, wait_ready: bool = True) -> bool:
        """
        Launch an application.
        
        Args:
            app_name: Application name (e.g., "Chrome", "Notes")
            wait_ready: Wait for app to be ready
        
        Returns:
            True if launched successfully
        """
        self.logger.info(f"Launching {app_name}...")
        
        # Get bundle ID
        bundle_id = self._get_bundle_id(app_name)
        
        if bundle_id:
            success = self._launch_by_bundle_id(bundle_id)
        else:
            success = self._launch_by_name(app_name)
        
        if success and wait_ready:
            time.sleep(1.0)
            self.activate(app_name)
            time.sleep(0.5)
        
        return success
    
    def activate(self, app_name: str, timeout: float = 5.0) -> bool:
        """
        Activate (focus) an application.
        
        Args:
            app_name: Application name
            timeout: Max time to wait for activation
        
        Returns:
            True if activated successfully
        """
        self.logger.debug(f"Activating {app_name}...")
        
        running_apps = self.workspace.runningApplications()
        
        for app in running_apps:
            app_display_name = app.localizedName()
            
            if self._matches_app_name(app_display_name, app_name):
                # NSApplicationActivateIgnoringOtherApps
                success = app.activateWithOptions_(1 << 1)
                
                if success:
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        if self.is_active(app_name):
                            self.logger.debug(f"Activated {app_name}")
                            return True
                        time.sleep(0.1)
        
        self.logger.warning(f"Failed to activate {app_name}")
        return False
    
    def is_running(self, app_name: str) -> bool:
        """Check if application is running."""
        running_apps = self.workspace.runningApplications()
        
        for app in running_apps:
            if self._matches_app_name(app.localizedName(), app_name):
                return True
        
        return False
    
    def is_active(self, app_name: str) -> bool:
        """Check if application is currently active (focused)."""
        active_app = self.workspace.activeApplication()
        if not active_app:
            return False
        
        active_name = active_app.get('NSApplicationName', '')
        return self._matches_app_name(active_name, app_name)
    
    def get_active_app(self) -> str:
        """Get name of currently active application."""
        active_app = self.workspace.activeApplication()
        if active_app:
            return active_app.get('NSApplicationName', 'Unknown')
        return 'Unknown'
    
    def wait_for_app(self, app_name: str, timeout: float = 10.0) -> bool:
        """
        Wait for app to be running and ready.
        
        Args:
            app_name: Application name
            timeout: Max time to wait
        
        Returns:
            True if app is ready
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_running(app_name):
                time.sleep(0.5)
                return True
            time.sleep(0.2)
        
        return False
    
    def ensure_active(self, app_name: str, launch_if_needed: bool = True) -> bool:
        """
        Ensure the specified app is active.
        
        Args:
            app_name: Target application name
            launch_if_needed: Launch if not running
        
        Returns:
            True if app is now active
        """
        if self.is_active(app_name):
            return True
        
        if not self.is_running(app_name):
            if launch_if_needed:
                if not self.launch(app_name):
                    return False
            else:
                self.logger.error(f"{app_name} is not running")
                return False
        
        return self.activate(app_name)
    
    def quit_app(self, app_name: str) -> bool:
        """Quit an application gracefully."""
        running_apps = self.workspace.runningApplications()
        
        for app in running_apps:
            if self._matches_app_name(app.localizedName(), app_name):
                return app.terminate()
        
        return False
    
    def _get_bundle_id(self, app_name: str) -> Optional[str]:
        """Get bundle ID for app name."""
        app_lower = app_name.lower()
        
        for key, bundle_id in self.BUNDLE_IDS.items():
            if key in app_lower:
                return bundle_id
        
        return None
    
    def _launch_by_bundle_id(self, bundle_id: str) -> bool:
        """Launch app by bundle ID."""
        try:
            # NSWorkspaceLaunchAsync = 1 << 6
            success = self.workspace.launchAppWithBundleIdentifier_options_additionalEventParamDescriptor_launchIdentifier_(
                bundle_id,
                1 << 6,
                None,
                None
            )[0]
            
            return success
        except Exception as e:
            self.logger.warning(f"Failed to launch by bundle ID: {e}")
            return False
    
    def _launch_by_name(self, app_name: str) -> bool:
        """Launch app by name using 'open' command."""
        try:
            result = subprocess.run(
                ['open', '-a', app_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return result.returncode == 0
        except Exception as e:
            self.logger.warning(f"Failed to launch by name: {e}")
            return False
    
    def _matches_app_name(self, app_display_name: str, target_name: str) -> bool:
        """Check if app names match (case-insensitive, partial match)."""
        if not app_display_name or not target_name:
            return False
        
        app_lower = app_display_name.lower().replace(' browser', '').replace('.app', '')
        target_lower = target_name.lower().replace(' browser', '').replace('.app', '')
        
        return target_lower in app_lower or app_lower in target_lower