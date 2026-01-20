"""Rate limiting utilities for API calls.

Prevents hitting rate limits on external APIs (OpenAI, Gemini, etc.)
by throttling requests when approaching limits.
"""
import time
from collections import deque
from threading import Lock
from typing import Optional, Dict
from dataclasses import dataclass
from functools import wraps

from src.utils.logger import setup_logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    calls_per_minute: int = 60
    calls_per_hour: int = 1000
    min_interval_seconds: float = 0.0  # Minimum time between calls


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.
    
    Features:
    - Per-minute and per-hour limits
    - Minimum interval between calls
    - Automatic waiting when limit reached
    - Statistics tracking
    
    Usage:
        limiter = RateLimiter(calls_per_minute=30)
        
        # Before each API call:
        limiter.acquire()  # Will wait if rate limit reached
        api_call()
        
        # Or use as decorator:
        @limiter.limit
        def api_call():
            ...
    """
    
    def __init__(
        self,
        calls_per_minute: int = 60,
        calls_per_hour: int = 1000,
        min_interval_seconds: float = 0.0,
        name: str = "default"
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls allowed per minute
            calls_per_hour: Maximum calls allowed per hour
            min_interval_seconds: Minimum seconds between consecutive calls
            name: Name for logging purposes
        """
        self.name = name
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.min_interval_seconds = min_interval_seconds
        
        self._minute_window = deque()  # Timestamps of calls in last minute
        self._hour_window = deque()    # Timestamps of calls in last hour
        self._last_call: Optional[float] = None
        self._lock = Lock()
        
        # Statistics
        self._total_calls = 0
        self._total_wait_time = 0.0
        self._times_throttled = 0
        
        self.logger = setup_logger(f"RateLimiter:{name}")
    
    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make an API call.
        
        Blocks until the rate limit allows a call, or timeout is reached.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if acquired, False if timeout reached
        """
        start_wait = time.time()
        
        while True:
            wait_time = self._calculate_wait_time()
            
            if wait_time <= 0:
                # Can proceed immediately
                self._record_call()
                return True
            
            # Check timeout
            elapsed = time.time() - start_wait
            if elapsed + wait_time > timeout:
                self.logger.warning(f"Rate limit timeout after {elapsed:.2f}s")
                return False
            
            # Wait and retry
            self.logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
            self._times_throttled += 1
            self._total_wait_time += wait_time
            time.sleep(wait_time)
    
    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before next call is allowed."""
        with self._lock:
            now = time.time()
            wait_times = []
            
            # Clean old entries and check per-minute limit
            minute_ago = now - 60.0
            while self._minute_window and self._minute_window[0] < minute_ago:
                self._minute_window.popleft()
            
            if len(self._minute_window) >= self.calls_per_minute:
                # Need to wait until oldest call falls out of window
                oldest = self._minute_window[0]
                wait_times.append(oldest - minute_ago)
            
            # Clean old entries and check per-hour limit
            hour_ago = now - 3600.0
            while self._hour_window and self._hour_window[0] < hour_ago:
                self._hour_window.popleft()
            
            if len(self._hour_window) >= self.calls_per_hour:
                oldest = self._hour_window[0]
                wait_times.append(oldest - hour_ago)
            
            # Check minimum interval
            if self._last_call and self.min_interval_seconds > 0:
                time_since_last = now - self._last_call
                if time_since_last < self.min_interval_seconds:
                    wait_times.append(self.min_interval_seconds - time_since_last)
            
            return max(wait_times) if wait_times else 0.0
    
    def _record_call(self):
        """Record that a call was made."""
        with self._lock:
            now = time.time()
            self._minute_window.append(now)
            self._hour_window.append(now)
            self._last_call = now
            self._total_calls += 1
    
    def try_acquire(self) -> bool:
        """
        Try to acquire permission without waiting.
        
        Returns:
            True if acquired, False if would need to wait
        """
        with self._lock:
            if self._calculate_wait_time() <= 0:
                self._record_call()
                return True
            return False
    
    def limit(self, func):
        """
        Decorator to rate limit a function.
        
        Usage:
            @limiter.limit
            def api_call():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "name": self.name,
                "total_calls": self._total_calls,
                "times_throttled": self._times_throttled,
                "total_wait_time_seconds": round(self._total_wait_time, 2),
                "calls_in_last_minute": len(self._minute_window),
                "calls_in_last_hour": len(self._hour_window),
                "limits": {
                    "per_minute": self.calls_per_minute,
                    "per_hour": self.calls_per_hour,
                    "min_interval": self.min_interval_seconds
                }
            }
    
    def reset(self):
        """Reset the rate limiter state and statistics."""
        with self._lock:
            self._minute_window.clear()
            self._hour_window.clear()
            self._last_call = None
            self._total_calls = 0
            self._total_wait_time = 0.0
            self._times_throttled = 0


class RateLimiterManager:
    """
    Manager for multiple rate limiters.
    
    Useful when you have multiple APIs with different rate limits.
    
    Usage:
        manager = RateLimiterManager()
        manager.register("gemini", calls_per_minute=30)
        manager.register("openai", calls_per_minute=60)
        
        manager.acquire("gemini")  # Uses gemini limiter
        manager.acquire("openai")  # Uses openai limiter
    """
    
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = Lock()
        self.logger = setup_logger("RateLimiterManager")
    
    def register(
        self,
        name: str,
        calls_per_minute: int = 60,
        calls_per_hour: int = 1000,
        min_interval_seconds: float = 0.0
    ) -> RateLimiter:
        """
        Register a new rate limiter.
        
        Args:
            name: Unique name for this limiter
            calls_per_minute: Max calls per minute
            calls_per_hour: Max calls per hour
            min_interval_seconds: Min seconds between calls
        
        Returns:
            The created RateLimiter
        """
        with self._lock:
            if name in self._limiters:
                self.logger.warning(f"Replacing existing limiter: {name}")
            
            limiter = RateLimiter(
                calls_per_minute=calls_per_minute,
                calls_per_hour=calls_per_hour,
                min_interval_seconds=min_interval_seconds,
                name=name
            )
            self._limiters[name] = limiter
            self.logger.info(f"Registered rate limiter: {name} ({calls_per_minute}/min)")
            return limiter
    
    def get(self, name: str) -> Optional[RateLimiter]:
        """Get a rate limiter by name."""
        return self._limiters.get(name)
    
    def acquire(self, name: str, timeout: float = 60.0) -> bool:
        """
        Acquire permission from a named rate limiter.
        
        Args:
            name: Name of the rate limiter
            timeout: Maximum wait time
        
        Returns:
            True if acquired, False if limiter not found or timeout
        """
        limiter = self._limiters.get(name)
        if not limiter:
            self.logger.warning(f"Unknown rate limiter: {name}")
            return True  # Allow call if limiter not configured
        
        return limiter.acquire(timeout)
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics from all rate limiters."""
        return {
            name: limiter.get_stats()
            for name, limiter in self._limiters.items()
        }


# Pre-configured limiters for common APIs
rate_limiters = RateLimiterManager()

# Register default limiters (conservative limits)
# These can be overridden by calling register() again
rate_limiters.register("gemini", calls_per_minute=30, min_interval_seconds=0.5)
rate_limiters.register("openai", calls_per_minute=60, min_interval_seconds=0.1)
rate_limiters.register("default", calls_per_minute=60)
