"""Pydantic models for graph-layer artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphBundle(BaseModel):
    """Aggregated graph payload across processed clips."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
