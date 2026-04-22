"""Build clip-level graph nodes from normalized clip data."""

from __future__ import annotations

from videographrag.clip_builder.models import ClipNode
from videographrag.normalization.schemas import NormalizedClipData


def build_clip_node(normalized_clip: NormalizedClipData) -> ClipNode:
    """Create a ClipNode placeholder from one normalized clip payload."""
    return ClipNode(clip_id=normalized_clip.clip_id)
