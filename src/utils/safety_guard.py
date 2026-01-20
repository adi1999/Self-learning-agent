"""Safety guardrails to prevent dangerous operations.

This module provides a safety layer that blocks potentially dangerous
operations during workflow execution, such as:
- System shutdown/restart commands
- File deletion on protected paths
- Dangerous terminal commands
- Browser settings that could cause data loss
"""
import re
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import setup_logger


class DangerLevel(Enum):
    """Severity level of detected danger."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class SafetyCheck:
    """Result of a safety check."""
    allowed: bool
    danger_level: DangerLevel
    reason: Optional[str] = None
    action_type: Optional[str] = None


class SafetyGuard:
    """
    Prevents execution of dangerous operations.
    
    Checks:
    - Keyboard shortcuts (system shutdown, force quit, etc.)
    - Typed commands (rm -rf, shutdown, etc.)
    - URLs (browser reset, clear data pages)
    - App-specific dangerous actions
    - File operations on protected paths
    
    Usage:
        from src.utils.safety_guard import safety_guard
        
        check = safety_guard.check_shortcut(("command", "option", "escape"))
        if not check.allowed:
            raise Exception(f"Blocked: {check.reason}")
    """
    
    # === BLOCKED SHORTCUTS (Always blocked) ===
    BLOCKED_SHORTCUTS: Set[Tuple[str, ...]] = {
        # System shutdown/restart/sleep
        ("command", "option", "control", "eject"),
        ("command", "control", "eject"),
        ("command", "option", "eject"),
        ("control", "command", "q"),  # Force quit all
        
        # Force quit (can interrupt important processes)
        ("command", "option", "escape"),
        
        # Logout
        ("command", "shift", "q"),
        
        # Lock screen (could lock out automation)
        ("command", "control", "q"),
    }
    
    # === WARNING SHORTCUTS (Blocked in strict mode) ===
    WARNING_SHORTCUTS = {
        # Delete operations
        ("command", "delete"): "delete_file",
        ("command", "backspace"): "delete_file",
        
        # Empty trash
        ("command", "shift", "delete"): "empty_trash",
        ("command", "shift", "backspace"): "empty_trash",
        
        # Close window/quit (could lose unsaved work)
        ("command", "q"): "quit_app",
        ("command", "w"): "close_window",
    }
    
    # === BLOCKED TYPED PATTERNS (Shell commands) ===
    BLOCKED_TYPE_PATTERNS = [
        # Destructive file operations
        r"^\s*sudo\s+rm\s+-rf\s+/",              # sudo rm -rf /
        r"^\s*rm\s+-rf\s+/\s*$",                  # rm -rf /
        r"^\s*rm\s+-rf\s+/[A-Za-z]",             # rm -rf /System, etc.
        r"^\s*rm\s+-rf\s+~/?$",                   # rm -rf ~ or ~/
        r"^\s*rm\s+-rf\s+\*",                     # rm -rf *
        r"^\s*rm\s+-rf\s+\.\.",                   # rm -rf ..
        
        # Disk operations
        r"^\s*mkfs\.",                            # mkfs.* (format disk)
        r"^\s*dd\s+if=.+of=/dev/",               # dd to disk device
        r">\s*/dev/sd[a-z]",                      # overwrite disk
        r">\s*/dev/nvme",                         # overwrite nvme
        r">\s*/dev/disk",                         # overwrite disk (macOS)
        
        # System control
        r"^\s*sudo\s+shutdown",                   # shutdown
        r"^\s*sudo\s+reboot",                     # reboot
        r"^\s*sudo\s+halt",                       # halt
        r"^\s*sudo\s+init\s+[06]",               # init 0 or init 6
        r"^\s*sudo\s+poweroff",                   # poweroff
        
        # Fork bomb and similar
        r":\(\)\s*\{.*\}\s*;?\s*:",              # fork bomb
        r"^\s*yes\s*\|",                          # yes | (infinite loop)
        
        # Permission attacks
        r"^\s*chmod\s+-R\s+777\s+/",             # dangerous chmod on root
        r"^\s*chmod\s+-R\s+000\s+/",             # remove all permissions
        r"^\s*chown\s+-R\s+.+\s+/\s*$",          # dangerous chown on root
        
        # macOS specific dangerous commands
        r"^\s*diskutil\s+eraseDisk",             # erase disk
        r"^\s*diskutil\s+partitionDisk",         # partition disk
        r"^\s*diskutil\s+secureErase",           # secure erase
        r"^\s*nvram\s+",                          # NVRAM modification
        r"^\s*csrutil\s+disable",                # disable System Integrity Protection
        r"^\s*sudo\s+systemsetup",               # system settings modification
        r"^\s*sudo\s+spctl\s+--master-disable",  # disable Gatekeeper
        
        # Network/Security modifications
        r"^\s*sudo\s+networksetup\s+-setwebproxy",    # proxy settings
        r"^\s*sudo\s+networksetup\s+-setsocksfirewall", # firewall
        r"^\s*security\s+delete-",                     # delete keychain items
        r"^\s*sudo\s+dscl\s+.+delete",                # delete directory service
        r"^\s*sudo\s+launchctl\s+unload.*com\.apple", # unload system services
        
        # Credential/sensitive data exposure
        r"cat\s+.+\.ssh/id_",                    # cat SSH keys
        r"cat\s+.+\.aws/credentials",           # cat AWS credentials
        r"cat\s+.+\.env",                        # cat environment files
    ]
    
    # === BLOCKED URLS ===
    BLOCKED_URL_PATTERNS = [
        # Chrome dangerous settings
        r"chrome://settings/clearBrowserData",
        r"chrome://settings/reset",
        r"chrome://settings/resetProfileSettings",
        
        # Firefox dangerous settings  
        r"about:config",
        r"about:preferences.*clear",
        
        # Edge dangerous settings
        r"edge://settings/reset",
        r"edge://settings/clearBrowserData",
        
        # Brave
        r"brave://settings/reset",
        r"brave://settings/clearBrowserData",
    ]
    
    # === PROTECTED PATHS (for file operations) ===
    PROTECTED_PATHS = [
        "/",
        "/System",
        "/Library", 
        "/usr",
        "/bin",
        "/sbin",
        "/etc",
        "/var",
        "/private",
        "/Applications",
        "/Users",
        "~",
        "~/Library",
        "~/Documents",
        "~/Desktop",
        "~/Downloads",
    ]
    
    # === BLOCKED APP CONTEXTS ===
    BLOCKED_APP_ACTIONS = {
        "Disk Utility": ["erase", "partition", "format", "restore"],
        "System Preferences": ["startup disk", "security & privacy", "users & groups"],
        "System Settings": ["startup disk", "privacy & security", "users & groups"],
        "Keychain Access": ["delete", "remove"],
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize safety guard.
        
        Args:
            strict_mode: If True, also blocks WARNING level actions.
                        If False, only blocks BLOCKED level actions.
        """
        self.logger = setup_logger("SafetyGuard")
        self.strict_mode = strict_mode
        
        # Pre-compile patterns for performance
        self._blocked_type_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKED_TYPE_PATTERNS
        ]
        self._blocked_url_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BLOCKED_URL_PATTERNS
        ]
        
        self.logger.info(f"SafetyGuard initialized (strict_mode={strict_mode})")
    
    def check_shortcut(self, keys: Tuple[str, ...]) -> SafetyCheck:
        """
        Check if a keyboard shortcut is safe to execute.
        
        Args:
            keys: Tuple of key names, e.g. ("command", "shift", "q")
        
        Returns:
            SafetyCheck with allowed=False if dangerous
        """
        # Normalize keys to lowercase
        keys_normalized = tuple(k.lower().strip() for k in keys)
        
        # Check absolutely blocked shortcuts
        if keys_normalized in self.BLOCKED_SHORTCUTS:
            self.logger.warning(f"🛑 BLOCKED shortcut: {'+'.join(keys)}")
            return SafetyCheck(
                allowed=False,
                danger_level=DangerLevel.BLOCKED,
                reason=f"Dangerous shortcut blocked: {'+'.join(keys)}",
                action_type="shortcut"
            )
        
        # Check warning-level shortcuts
        for warning_keys, action_name in self.WARNING_SHORTCUTS.items():
            if keys_normalized == warning_keys:
                if self.strict_mode:
                    self.logger.warning(f"⚠️ BLOCKED shortcut (strict): {'+'.join(keys)} ({action_name})")
                    return SafetyCheck(
                        allowed=False,
                        danger_level=DangerLevel.WARNING,
                        reason=f"Potentially dangerous shortcut ({action_name}): {'+'.join(keys)}",
                        action_type="shortcut"
                    )
                else:
                    self.logger.debug(f"⚠️ Allowing warning shortcut: {'+'.join(keys)} ({action_name})")
        
        return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
    
    def check_typed_text(self, text: str, app_name: str = "") -> SafetyCheck:
        """
        Check if typed text contains dangerous commands.
        
        Only applies strict checking to terminal-like applications.
        
        Args:
            text: The text being typed
            app_name: Name of the target application
        
        Returns:
            SafetyCheck with allowed=False if dangerous command detected
        """
        # Only check in terminal-like apps
        terminal_apps = {
            "Terminal", "iTerm", "iTerm2", "Hyper", "Alacritty", 
            "kitty", "Warp", "Tabby", "Terminus", "Console"
        }
        
        # Also check if it's being typed into a shell-like context
        is_terminal_context = (
            app_name in terminal_apps or 
            "terminal" in app_name.lower() or
            "console" in app_name.lower() or
            "shell" in app_name.lower()
        )
        
        if not is_terminal_context:
            return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
        
        # Check against blocked patterns
        for pattern in self._blocked_type_patterns:
            if pattern.search(text):
                self.logger.warning(f"🛑 BLOCKED command: {text[:60]}...")
                return SafetyCheck(
                    allowed=False,
                    danger_level=DangerLevel.BLOCKED,
                    reason=f"Dangerous command blocked: {text[:50]}...",
                    action_type="type"
                )
        
        return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
    
    def check_url(self, url: str) -> SafetyCheck:
        """
        Check if URL navigation is safe.
        
        Args:
            url: The URL being navigated to
        
        Returns:
            SafetyCheck with allowed=False if dangerous URL
        """
        if not url:
            return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
        
        for pattern in self._blocked_url_patterns:
            if pattern.search(url):
                self.logger.warning(f"🛑 BLOCKED URL: {url}")
                return SafetyCheck(
                    allowed=False,
                    danger_level=DangerLevel.BLOCKED,
                    reason=f"Dangerous URL blocked: {url}",
                    action_type="navigate"
                )
        
        return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
    
    def check_app_action(self, app_name: str, action_description: str) -> SafetyCheck:
        """
        Check if an action in a specific app is safe.
        
        Args:
            app_name: Name of the application
            action_description: Description of what's being done
        
        Returns:
            SafetyCheck with allowed=False if dangerous action in sensitive app
        """
        blocked_keywords = self.BLOCKED_APP_ACTIONS.get(app_name, [])
        
        for keyword in blocked_keywords:
            if keyword.lower() in action_description.lower():
                self.logger.warning(f"🛑 BLOCKED action in {app_name}: {action_description}")
                return SafetyCheck(
                    allowed=False,
                    danger_level=DangerLevel.BLOCKED,
                    reason=f"Dangerous action blocked in {app_name}: {action_description}",
                    action_type="app_action"
                )
        
        return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
    
    def check_file_operation(self, path: str, operation: str) -> SafetyCheck:
        """
        Check if a file operation is safe.
        
        Args:
            path: File/directory path
            operation: Operation type (delete, remove, trash, etc.)
        
        Returns:
            SafetyCheck with allowed=False if operation on protected path
        """
        dangerous_operations = {"delete", "remove", "rm", "trash", "unlink", "rmdir"}
        
        if operation.lower() not in dangerous_operations:
            return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
        
        import os
        expanded_path = os.path.expanduser(path)
        abs_path = os.path.abspath(expanded_path)
        
        for protected in self.PROTECTED_PATHS:
            protected_expanded = os.path.expanduser(protected)
            protected_abs = os.path.abspath(protected_expanded) if not protected.startswith("/") else protected_expanded
            
            # Block if trying to delete the protected path itself
            if abs_path == protected_abs:
                self.logger.warning(f"🛑 BLOCKED {operation} on protected path: {path}")
                return SafetyCheck(
                    allowed=False,
                    danger_level=DangerLevel.BLOCKED,
                    reason=f"Cannot {operation} protected path: {path}",
                    action_type="file_operation"
                )
            
            # Block if trying to delete root-level system directories
            if protected in ["/", "/System", "/Library", "/usr", "/bin", "/sbin", "/etc", "/var", "/private"]:
                if abs_path.startswith(protected_abs + "/") or abs_path == protected_abs:
                    # Allow deleting files deep in user directories
                    if protected in ["~", "~/Library", "~/Documents", "~/Desktop", "~/Downloads"]:
                        # Only block top-level deletion of these
                        if abs_path == protected_abs:
                            self.logger.warning(f"🛑 BLOCKED {operation} on user directory: {path}")
                            return SafetyCheck(
                                allowed=False,
                                danger_level=DangerLevel.BLOCKED,
                                reason=f"Cannot {operation} user directory: {path}",
                                action_type="file_operation"
                            )
                    else:
                        self.logger.warning(f"🛑 BLOCKED {operation} in system path: {path}")
                        return SafetyCheck(
                            allowed=False,
                            danger_level=DangerLevel.BLOCKED,
                            reason=f"Cannot {operation} in system path: {path}",
                            action_type="file_operation"
                        )
        
        return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)
    
    def is_safe(
        self,
        shortcut: Optional[Tuple[str, ...]] = None,
        typed_text: Optional[str] = None,
        app_name: str = "",
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        file_operation: Optional[str] = None
    ) -> SafetyCheck:
        """
        Convenience method to check multiple safety conditions at once.
        
        Returns the first failed check, or a SAFE check if all pass.
        """
        if shortcut:
            check = self.check_shortcut(shortcut)
            if not check.allowed:
                return check
        
        if typed_text:
            check = self.check_typed_text(typed_text, app_name)
            if not check.allowed:
                return check
        
        if url:
            check = self.check_url(url)
            if not check.allowed:
                return check
        
        if file_path and file_operation:
            check = self.check_file_operation(file_path, file_operation)
            if not check.allowed:
                return check
        
        return SafetyCheck(allowed=True, danger_level=DangerLevel.SAFE)


# Global instance (non-strict by default for backward compatibility)
safety_guard = SafetyGuard(strict_mode=False)
