"""Backward-compatible alias for the project console logger.

Prefer importing `get_console_logger` from `src.logger`.
"""

from src.logger import get_console_logger

__all__ = ["get_console_logger"]
