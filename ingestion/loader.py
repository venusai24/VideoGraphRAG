"""Load raw per-clip JSON files from Azure Video Indexer outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from videographrag.normalization.schemas import RawClipInput

REQUIRED_FILES = {
    "keywords.json",
    "ocr.json",
    "rag_chunks.json",
    "raw_insights.json",
    "scenes.json",
    "transcript.json",
}


def iter_clip_dirs(root_dir: Path) -> Iterator[Path]:
    """Yield clip directories that contain the required JSON files."""
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and REQUIRED_FILES.issubset({p.name for p in child.iterdir()}):
            yield child


def load_clip_bundle(clip_dir: Path) -> RawClipInput:
    """Load all required JSON files for one clip folder into a raw schema."""
    payload = {}
    for filename in REQUIRED_FILES:
        with (clip_dir / filename).open("r", encoding="utf-8") as f:
            payload[filename.replace(".json", "")] = json.load(f)
    return RawClipInput(clip_id=clip_dir.name, **payload)
