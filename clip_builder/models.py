"""Pydantic models for clip-layer graph objects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClipNode(BaseModel):
    """Graph-ready representation of one processed video clip."""

    clip_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
