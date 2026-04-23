"""Normalization stage for mapping raw clip payloads into unified schemas."""

from __future__ import annotations

from videographrag.normalization.schemas import NormalizedClipData, RawClipInput


def normalize_clip(raw_clip: RawClipInput) -> NormalizedClipData:
    """Normalize one clip's raw payload into a stable intermediate contract."""
    return NormalizedClipData(clip_id=raw_clip.clip_id)
