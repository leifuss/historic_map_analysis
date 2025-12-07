"""
Utility functions for PDF processing.
"""

import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str = "config/processing_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('log_file', 'output/processing.log')
    console_output = log_config.get('console_output', True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def format_bytes(bytes: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_processing_time(page_count: int, time_per_page: float = 2.0) -> Dict[str, Any]:
    """
    Estimate processing time based on page count.

    Args:
        page_count: Number of pages
        time_per_page: Average seconds per page

    Returns:
        Dictionary with time estimates
    """
    total_seconds = page_count * time_per_page

    return {
        'total_seconds': total_seconds,
        'formatted': format_duration(total_seconds),
        'pages': page_count,
        'time_per_page': time_per_page
    }


def create_output_filename(
    prefix: str,
    page_num: int,
    extension: str,
    total_pages: int
) -> str:
    """
    Create standardized output filename.

    Args:
        prefix: Filename prefix
        page_num: Page number
        extension: File extension (without dot)
        total_pages: Total number of pages (for zero-padding)

    Returns:
        Formatted filename
    """
    # Determine padding based on total pages
    if total_pages < 100:
        padding = 2
    elif total_pages < 1000:
        padding = 3
    else:
        padding = 4

    return f"{prefix}_{page_num:0{padding}d}.{extension}"


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.

    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ProgressTracker:
    """Simple progress tracker for batch operations."""

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.

        Returns:
            Dictionary with progress data
        """
        percent = (self.current / self.total * 100) if self.total > 0 else 0
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if self.current > 0:
            time_per_item = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = remaining_items * time_per_item
        else:
            eta_seconds = 0

        return {
            'current': self.current,
            'total': self.total,
            'percent': round(percent, 1),
            'elapsed': format_duration(elapsed),
            'eta': format_duration(eta_seconds),
            'description': self.description
        }

    def print_progress(self):
        """Print progress to console."""
        progress = self.get_progress()
        print(
            f"\r{progress['description']}: {progress['current']}/{progress['total']} "
            f"({progress['percent']}%) - "
            f"Elapsed: {progress['elapsed']} - "
            f"ETA: {progress['eta']}",
            end='',
            flush=True
        )

    def complete(self):
        """Mark as complete and print final message."""
        self.current = self.total
        progress = self.get_progress()
        print(
            f"\n{progress['description']} complete! "
            f"Total time: {progress['elapsed']}"
        )
