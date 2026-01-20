"""Main session recorder - orchestrates all capture components."""
import time
import signal
from pathlib import Path
from typing import Optional, List
from threading import Lock
from queue import Queue, Empty

from src.models.session_artifact import (
    SessionArtifact, TimelineEvent, InputEvent, ElementInfo, NavigationOutcome
)
from urllib.parse import urlparse
from src.observer.input_capture import InputCapture
from src.observer.screen_capture import ScreenCapture
from src.observer.voice_capture import VoiceCapture
from src.observer.window_manager import WindowManager
from src.observer.browser_capture import BrowserCapture
from src.observer.desktop_capture import DesktopCapture
from src.utils.logger import setup_logger
from src.utils.config import config


class SessionRecorder:
    """
    Orchestrates recording of user workflow.
    
    Coordinates:
    - Input capture (keyboard/mouse)
    - Screen capture (event-driven screenshots)
    - Voice capture (raw audio - transcription in compile phase)
    - Browser capture (Playwright for DOM)
    - Desktop capture (macapptree for accessibility)
    
    IMPORTANT: Uses queue-based communication to avoid threading issues.
    InputCapture runs in a separate thread, but Playwright must only be
    called from the main thread. Events are queued and processed in the
    main recording loop.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        use_browser: bool = True,
        initial_url: Optional[str] = None,
        enable_voice: bool = True
    ):
        """
        Initialize session recorder.
        
        Args:
            output_dir: Where to save session artifacts
            use_browser: Launch Playwright browser for recording
            initial_url: Initial URL for browser
            enable_voice: Enable voice recording
        """
        self.logger = setup_logger("SessionRecorder")
        self.output_dir = output_dir or config.sessions_dir
        self.use_browser = use_browser
        self.initial_url = initial_url or config.browser_default_url
        self.enable_voice = enable_voice and config.voice_enabled
        
        # Initialize session
        self.session = SessionArtifact()
        self.session_dir: Optional[Path] = None
        
        # Components
        self.input_capture: Optional[InputCapture] = None
        self.screen_capture: Optional[ScreenCapture] = None
        self.voice_capture: Optional[VoiceCapture] = None
        self.window_manager = WindowManager()
        self.browser_capture: Optional[BrowserCapture] = None
        self.desktop_capture = DesktopCapture()
        
        # State
        self.is_recording = False
        self.start_time: Optional[float] = None
        self._last_app: Optional[str] = None
        self._last_url: Optional[str] = None
        
        # ============================================================
        # THREAD-SAFE EVENT QUEUE
        # ============================================================
        # InputCapture callback puts events here (from CFRunLoop thread)
        # Main thread processes them (safe to call Playwright)
        self._event_queue: Queue = Queue()
        
        # Pending events for current timeline event
        self._pending_events: List[InputEvent] = []
        self._events_lock = Lock()
        
        # Signal handling
        self._original_sigint = None
    
    def _get_timestamp(self) -> float:
        """Get current timestamp relative to recording start."""
        if not self.start_time:
            return 0.0
        return time.time() - self.start_time
    
    def _on_input_event(self, event: InputEvent):
        """
        Callback for input events from InputCapture.
        
        NOTE: This runs in the CFRunLoop thread!
        Do NOT call Playwright here. Just queue the event.
        """
        # Just put the event in the queue - main thread will process it
        self._event_queue.put(event)
    
    def _process_queued_events(self):
        """
        Process all queued events from the main thread.
        
        This is safe to call Playwright from here.
        """
        while True:
            try:
                event = self._event_queue.get_nowait()
                self._handle_input_event(event)
            except Empty:
                break
    
    def _handle_input_event(self, event: InputEvent):
        """
        Handle a single input event (called from main thread).
        
        Safe to call Playwright here.
        """
        with self._events_lock:
            self._pending_events.append(event)
        
        # Determine if this event should trigger a timeline event
        trigger = self._get_trigger_type(event)
        
        if trigger:
            self._create_timeline_event(trigger, event)
    
    def _get_trigger_type(self, event: InputEvent) -> Optional[str]:
        """Determine what trigger type (if any) this event represents."""
        if event.type == "mouse_click":
            return "click"
        
        if event.type == "keyboard":
            if event.key == "return":
                # Check if this is a submit (single-line input) or just Enter
                if self.browser_capture and self.browser_capture.page:
                    focused = self.browser_capture.get_focused_element()
                    if self.browser_capture.is_single_line_input(focused):
                        return "submit"
                # For desktop, treat Enter as potential submit
                elif self.window_manager.get_platform() == "desktop":
                    return "submit"
            
            # Typed text that was flushed
            if event.text and event.flush_reason:
                if event.flush_reason in ["submit", "click"]:
                    return event.flush_reason
        
        if event.type == "keyboard_shortcut":
            if event.shortcut == "app_switch":
                return "app_switch"
            if event.shortcut == "save":
                return "save"
            if event.shortcut == "copy":
                return "copy"  # Important for extraction detection
        
        return None
    
    def _create_timeline_event(self, trigger: str, triggering_event: InputEvent):
        """Create a timeline event from accumulated input."""
        timestamp = self._get_timestamp()
        
        # Get current context
        app_name = self.window_manager.get_active_app()
        window_title = self.window_manager.get_active_window_title()
        platform = self.window_manager.get_platform(app_name)
        
        # Check for app switch
        if self._last_app and app_name != self._last_app:
            trigger = "app_switch"
            # Flush any pending typing
            if self.input_capture:
                self.input_capture.force_flush("app_switch")
        
        # Get URL for browser
        url = None
        if platform == "browser" and self.browser_capture:
            url = self.browser_capture.get_current_url()
            
            # Check for navigation
            if self._last_url and url != self._last_url:
                trigger = "page_load"
        
        # Take screenshot (especially important for copy events - used for extraction analysis)
        screenshot_path = None
        if self.screen_capture:
            screenshot_path = self.screen_capture.capture(trigger, timestamp)
        
        # Get element info for clicks (NOW SAFE - we're in main thread!)
        element_info = None
        if triggering_event.type == "mouse_click" and triggering_event.x and triggering_event.y:
            element_info = self._capture_element_info(
                app_name, platform,
                triggering_event.x, triggering_event.y
            )
            triggering_event.element_info = element_info
        
        # Collect pending events
        with self._events_lock:
            events = self._pending_events.copy()
            self._pending_events.clear()
        
        # Capture navigation outcome for clicks in browser
        nav_outcome = None
        if trigger == "click" and platform == "browser":
            nav_outcome = self._capture_navigation_outcome(platform, url, wait_ms=1500)
            if nav_outcome and nav_outcome.domain_changed:
                self.logger.info(
                    f"[{timestamp:.1f}s] Navigation: {nav_outcome.domain_before} → {nav_outcome.domain_after}"
                )
        
        # Create timeline event
        timeline_event = TimelineEvent(
            timestamp=timestamp,
            active_app=app_name,
            window_title=window_title,
            platform=platform,
            url=url,
            screenshot_path=screenshot_path,
            input_events=events,
            boundary_reason=trigger,
            navigation_outcome=nav_outcome
        )
        
        self.session.add_event(timeline_event)
        
        # Update state
        self._last_app = app_name
        self._last_url = url
        
        # Log with special note for copy events
        if trigger == "copy" and triggering_event.clipboard_content:
            preview = triggering_event.clipboard_content[:50]
            self.logger.info(f"[{timestamp:.1f}s] {trigger}: Copied '{preview}...'")
        else:
            self.logger.debug(f"[{timestamp:.1f}s] {trigger}: {app_name} - {len(events)} events")
    
    def _capture_element_info(
        self, 
        app_name: str, 
        platform: str, 
        x: int, 
        y: int
    ) -> Optional[ElementInfo]:
        """
        Capture element info at coordinates.
        
        NOTE: This must be called from the main thread only!
        """
        if platform == "browser" and self.browser_capture and self.browser_capture.page:
            # Convert screen coords to viewport coords
            vx, vy = self.browser_capture.screen_to_viewport_coords(x, y)
            return self.browser_capture.get_element_at_point(vx, vy)
        
        elif platform == "desktop" and self.desktop_capture.is_available:
            return self.desktop_capture.capture_element_at_click(app_name, x, y)
        
        return None
    
    def _extract_domain(self, url: Optional[str]) -> Optional[str]:
        """Extract domain from URL."""
        if not url:
            return None
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
            return domain if domain else None
        except:
            return None
    
    def _capture_navigation_outcome(
        self,
        platform: str,
        url_before: Optional[str],
        wait_ms: int = 1500
    ) -> Optional[NavigationOutcome]:
        """
        Capture navigation outcome after a click.
        
        Waits briefly for navigation to complete, then captures the new URL.
        """
        if platform != "browser" or not self.browser_capture or not self.browser_capture.page:
            return None
        
        # Wait for navigation to potentially complete
        try:
            # Short wait for immediate navigation
            self.browser_capture.page.wait_for_load_state("domcontentloaded", timeout=wait_ms)
        except:
            pass
        
        time.sleep(0.1)  # Small additional wait
        
        url_after = self.browser_capture.get_current_url()
        domain_before = self._extract_domain(url_before)
        domain_after = self._extract_domain(url_after)
        
        # Determine navigation type
        nav_type = "same_page"
        domain_changed = False
        
        if url_before and url_after and url_after != url_before:
            if domain_before != domain_after:
                nav_type = "cross_domain"
                domain_changed = True
            else:
                nav_type = "same_domain"
        
        # Get page title
        page_title = None
        try:
            page_title = self.browser_capture.page.title()
        except:
            pass
        
        return NavigationOutcome(
            url_before=url_before,
            url_after=url_after,
            domain_before=domain_before,
            domain_after=domain_after,
            domain_changed=domain_changed,
            page_title_after=page_title,
            navigation_type=nav_type,
            wait_time_ms=wait_ms
        )
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n")
        self.logger.info("Stopping recording (Ctrl+C)...")
        self.stop()
    
    def start(self) -> SessionArtifact:
        """
        Start recording session.
        
        Returns:
            SessionArtifact after recording completes
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Workflow Recording")
        self.logger.info("=" * 60)
        
        # Create session directory
        self.session_dir = self.session.save(self.output_dir)
        screenshots_dir = self.session_dir / "screenshots"
        voice_dir = self.session_dir / "voice"
        
        # Initialize components
        self.screen_capture = ScreenCapture(screenshots_dir)
        self.input_capture = InputCapture(callback=self._on_input_event)
        
        if self.enable_voice:
            self.voice_capture = VoiceCapture(voice_dir)
        
        # Launch browser if requested
        if self.use_browser:
            self.logger.info("Launching browser...")
            self.browser_capture = BrowserCapture()
            self.browser_capture.launch(url=self.initial_url, headless=False)
            self.session.browser_used = "chromium"
            self.session.initial_url = self.initial_url
        
        # Set up signal handler
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start recording
        self.is_recording = True
        self.start_time = time.time()
        
        self.input_capture.start()
        
        if self.voice_capture:
            if self.voice_capture.start():
                self.logger.info("Voice recording enabled (transcription at compile time)")
            else:
                self.logger.warning("Voice recording failed to start")
        
        self.logger.info("")
        self.logger.info("Recording started!")
        self.logger.info("Perform your workflow. Press Ctrl+C when done.")
        self.logger.info(f"Session: {self.session.session_id}")
        self.logger.info("=" * 60)
        
        try:
            self._recording_loop()
        except KeyboardInterrupt:
            pass
        finally:
            return self.stop()
    
    def _recording_loop(self):
        """
        Main recording loop.
        
        This runs in the main thread, so it's safe to call Playwright here.
        """
        backup_interval = config.backup_screenshot_interval
        last_backup = 0.0
        
        while self.is_recording:
            timestamp = self._get_timestamp()
            
            # ============================================================
            # PROCESS QUEUED EVENTS (main thread - safe for Playwright)
            # ============================================================
            self._process_queued_events()
            
            # Check for app changes
            current_app = self.window_manager.get_active_app()
            if self._last_app and current_app != self._last_app:
                self.input_capture.force_flush("app_switch")
                self._create_timeline_event("app_switch", InputEvent(
                    type="keyboard_shortcut",
                    timestamp=timestamp,
                    shortcut="app_switch"
                ))
            self._last_app = current_app
            
            # Backup screenshot at intervals
            if timestamp - last_backup > backup_interval:
                if self.screen_capture:
                    self.screen_capture.capture("backup", timestamp)
                last_backup = timestamp
            
            time.sleep(0.016)  # ~60 Hz loop for responsive event processing
    
    def stop(self) -> SessionArtifact:
        """Stop recording and finalize session."""
        if not self.is_recording:
            return self.session
        
        self.is_recording = False
        
        # Process any remaining queued events
        self._process_queued_events()
        
        # Restore signal handler
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        
        # Stop components
        if self.input_capture:
            self.input_capture.stop()
        
        if self.screen_capture:
            self.screen_capture.close()
        
        # Stop voice recording (save audio, NO transcription)
        if self.voice_capture:
            audio_path = self.voice_capture.stop()
            if audio_path:
                # Store relative path for portability
                self.session.voice_audio_path = f"voice/{audio_path.name}"
                self.logger.info(f"Voice audio saved: {self.session.voice_audio_path}")
                self.logger.info("(Transcription will happen during compile phase)")
        
        # Close browser
        if self.browser_capture:
            self.browser_capture.close()
        
        # Finalize session
        self.session.finalize()
        self.session.save(self.output_dir)
        
        # Summary
        duration = self.session.duration()
        event_count = len(self.session.timeline)
        copy_count = len(self.session.get_copy_events())
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Recording Complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration:.1f} seconds")
        self.logger.info(f"Timeline events: {event_count}")
        self.logger.info(f"Copy events (extraction candidates): {copy_count}")
        self.logger.info(f"Session saved: {self.session_dir}")
        
        if self.session.voice_audio_path:
            self.logger.info(f"Voice audio: {self.session.voice_audio_path}")
        
        self.logger.info("=" * 60)
        
        return self.session