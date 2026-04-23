"""Pydantic schemas for clip-level raw and normalized data contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawClipInput(BaseModel):
    """Raw payload loaded from one clip folder's six JSON files."""

    clip_id: str
    keywords: list[dict[str, Any]] = Field(default_factory=list)
    ocr: list[dict[str, Any]] = Field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = Field(default_factory=list)
    raw_insights: dict[str, Any] = Field(default_factory=dict)
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    transcript: list[dict[str, Any]] = Field(default_factory=list)


class NormalizedClipData(BaseModel):
    """Unified intermediate schema used by downstream pipeline modules."""

    clip_id: str
    keywords: list[dict[str, Any]] = Field(default_factory=list)
    text_spans: list[dict[str, Any]] = Field(default_factory=list)
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    transcript_segments: list[dict[str, Any]] = Field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
