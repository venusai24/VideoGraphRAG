"""General helper placeholders shared by pipeline modules."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
