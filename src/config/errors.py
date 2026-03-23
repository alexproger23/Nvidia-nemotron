from __future__ import annotations

from pathlib import Path


class ConfigError(Exception):
    """Base error for config loading and validation."""


class ConfigNotFoundError(ConfigError):
    """Raised when an expected config file does not exist."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Config file was not found: {path}")
        self.path = path


class ConfigValidationError(ConfigError):
    """Raised when a config file has invalid structure or values."""

    def __init__(self, context: str, message: str) -> None:
        super().__init__(f"{context}: {message}")
        self.context = context
        self.message = message
