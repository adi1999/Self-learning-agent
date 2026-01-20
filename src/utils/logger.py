"""Logging utilities for PbD system."""
import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        structured: Use structured (JSON) logging
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if structured:
        console_format = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        console_handler.setFormatter(logging.Formatter(console_format))
    else:
        console_format = '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'
        console_handler.setFormatter(ColoredFormatter(console_format, datefmt='%H:%M:%S'))
    
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
    
    return logger


class StepLogger:
    """Context manager for logging step execution with timing."""
    
    def __init__(self, logger: logging.Logger, step_name: str, step_num: int = 0):
        self.logger = logger
        self.step_name = step_name
        self.step_num = step_num
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"[Step {self.step_num}] Starting: {self.step_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(f"[Step {self.step_num}] Failed: {self.step_name} ({duration:.2f}s) - {exc_val}")
        else:
            self.logger.info(f"[Step {self.step_num}] Completed: {self.step_name} ({duration:.2f}s)")
        
        return False  # Don't suppress exceptions