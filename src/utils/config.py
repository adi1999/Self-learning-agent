"""Configuration management for PbD workflow automation."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Central configuration for PbD system."""
    
    # =========================================================================
    # PATHS
    # =========================================================================
    artifacts_dir: Path = field(default_factory=lambda: Path.cwd() / "artifacts")
    
    @property
    def sessions_dir(self) -> Path:
        return self.artifacts_dir / "sessions"
    
    @property
    def recipes_dir(self) -> Path:
        return self.artifacts_dir / "recipes"
    
    # =========================================================================
    # INPUT CAPTURE SETTINGS
    # =========================================================================
    typing_buffer_size: int = 100  # Max chars to buffer before flush
    typing_idle_threshold: float = 0.5  # Seconds of idle to trigger flush
    
    # =========================================================================
    # SCREENSHOT SETTINGS
    # =========================================================================
    backup_screenshot_interval: float = 5.0  # Seconds between backup screenshots
    screenshot_triggers: list = field(default_factory=lambda: [
        "click", "submit", "app_switch", "copy", "save", "page_load", "backup"
    ])
    
    # =========================================================================
    # BROWSER SETTINGS
    # =========================================================================
    browser_type: str = "chromium"  # chromium, firefox, webkit
    browser_headless: bool = False
    browser_default_url: str = "https://www.google.com"
    
    # =========================================================================
    # VOICE SETTINGS
    # =========================================================================
    voice_enabled: bool = True
    voice_sample_rate: int = 16000
    
    # =========================================================================
    # OPENAI SETTINGS (GPT-4o for text analysis)
    # =========================================================================
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000
    
    # UPDATED: Use gpt-4o-transcribe instead of whisper-1
    whisper_model: str = "gpt-4o-transcribe"
    
    # =========================================================================
    # GOOGLE GEMINI SETTINGS (for visual understanding)
    # =========================================================================
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    gemini_model: str = "gemini-2.5-computer-use-preview-10-2025"
    gemini_use_for_extraction: bool = True
    gemini_use_as_fallback: bool = True
    gemini_use_for_validation: bool = False
    
    # =========================================================================
    # SEGMENTATION SETTINGS
    # =========================================================================
    idle_boundary_threshold: float = 2.0
    click_idle_threshold: float = 1.0
    backup_screenshot_interval: float = 5.0  # Seconds between backup screenshots
    
    # =========================================================================
    # EXECUTION SETTINGS
    # =========================================================================
    default_timeout: float = 10.0
    retry_limit: int = 2
    retry_delay: float = 1.0
    element_resolution_timeout: float = 5.0
    
    # =========================================================================
    # VALIDATION SETTINGS
    # =========================================================================
    validate_steps: bool = True
    abort_on_validation_failure: bool = False
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    structured_logs: bool = False
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # Override from environment
        if os.getenv("PBD_SESSIONS_DIR"):
            config.artifacts_dir = Path(os.getenv("PBD_SESSIONS_DIR")).parent
        
        if os.getenv("PBD_LOG_LEVEL"):
            config.log_level = os.getenv("PBD_LOG_LEVEL")
        
        if os.getenv("PBD_LLM_MODEL"):
            config.llm_model = os.getenv("PBD_LLM_MODEL")
        
        if os.getenv("PBD_GEMINI_MODEL"):
            config.gemini_model = os.getenv("PBD_GEMINI_MODEL")
        
        if os.getenv("PBD_BROWSER_HEADLESS"):
            config.browser_headless = os.getenv("PBD_BROWSER_HEADLESS").lower() == "true"
        
        if os.getenv("PBD_WHISPER_MODEL"):
            config.whisper_model = os.getenv("PBD_WHISPER_MODEL")
        
        return config
    
    def check_api_keys(self) -> dict:
        """Check which API keys are configured."""
        return {
            "openai": bool(self.openai_api_key),
            "google": bool(self.google_api_key),
        }
    
    def print_status(self):
        """Print configuration status."""
        keys = self.check_api_keys()
        print("\n=== PbD System Configuration ===")
        print(f"OpenAI API Key: {'✓ Set' if keys['openai'] else '✗ Not set'}")
        print(f"Google API Key: {'✓ Set' if keys['google'] else '✗ Not set'}")
        print(f"LLM Model: {self.llm_model}")
        print(f"Whisper Model: {self.whisper_model}")
        print(f"Gemini Model: {self.gemini_model}")
        print(f"Sessions Dir: {self.sessions_dir}")
        print(f"Recipes Dir: {self.recipes_dir}")
        print("================================\n")


# Global config instance
config = Config.from_env()