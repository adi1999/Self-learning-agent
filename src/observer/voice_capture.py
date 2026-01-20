"""Voice capture - records audio during workflow (transcription moved to compile phase)."""
import wave
import threading
from pathlib import Path
from typing import Optional
from src.utils.logger import setup_logger
from src.utils.config import config

# Try to import pyaudio, but make it optional
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class VoiceCapture:
    """
    Records audio during workflow demonstration.
    
    NOTE: Transcription is now done in the COMPILE phase, not here.
    This keeps the recording phase simple and fast.
    
    Voice narration provides:
    - Field labels ("this is the restaurant name")
    - Parameter hints ("searching for sushi")
    - Task context ("finding a restaurant for dinner")
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize voice capture.
        
        Args:
            output_dir: Directory to save audio files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("VoiceCapture")
        
        # Audio settings
        self.sample_rate = config.voice_sample_rate
        self.channels = 1
        self.chunk_size = 1024
        self.format = None  # Set in start() if pyaudio available
        
        # State
        self.is_recording = False
        self.audio_frames = []
        self._record_thread: Optional[threading.Thread] = None
        self._stream = None
        self._pyaudio = None
        
        # Output path
        self.audio_path: Optional[Path] = None
        
        if not PYAUDIO_AVAILABLE:
            self.logger.warning("PyAudio not available. Voice capture disabled.")
    
    @property
    def is_available(self) -> bool:
        """Check if voice capture is available."""
        return PYAUDIO_AVAILABLE and config.voice_enabled
    
    def start(self) -> bool:
        """
        Start recording audio.
        
        Returns:
            True if recording started successfully
        """
        if not self.is_available:
            self.logger.warning("Voice capture not available, skipping")
            return False
        
        try:
            self._pyaudio = pyaudio.PyAudio()
            self.format = pyaudio.paInt16
            
            self._stream = self._pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            self.audio_frames = []
            
            # Start recording thread
            self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self._record_thread.start()
            
            self.logger.info("Voice recording started")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start voice recording: {e}")
            return False
    
    def _record_loop(self):
        """Background recording loop."""
        while self.is_recording:
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_frames.append(data)
            except Exception as e:
                if self.is_recording:  # Only log if we didn't intentionally stop
                    self.logger.error(f"Recording error: {e}")
                break
    
    def stop(self) -> Optional[Path]:
        """
        Stop recording and save audio file.
        
        NOTE: Does NOT transcribe - that's done in compile phase now.
        
        Returns:
            Path to saved audio file, or None if failed
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Wait for recording thread
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        
        # Close stream
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        
        if self._pyaudio:
            self._pyaudio.terminate()
        
        # Save audio file
        if not self.audio_frames:
            self.logger.warning("No audio frames recorded")
            return None
        
        self.audio_path = self.output_dir / "voice_recording.wav"
        try:
            self._save_wav(self.audio_path)
            duration = len(self.audio_frames) * self.chunk_size / self.sample_rate
            self.logger.info(f"Saved voice recording: {self.audio_path} ({duration:.1f}s)")
            return self.audio_path
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return None
    
    def _save_wav(self, path: Path):
        """Save recorded audio as WAV file."""
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.audio_frames))
    
    def get_audio_path(self) -> Optional[Path]:
        """Get path to recorded audio file."""
        return self.audio_path