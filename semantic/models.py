"""Pydantic schemas for semantic extraction outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SemanticArtifact(BaseModel):
    """Container for semantic outputs derived from one clip."""

    clip_id: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
