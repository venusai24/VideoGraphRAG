"""Pipeline entrypoint for clip-folder-based GraphRAG processing.

Expected input layout:
- root_dir/
  - clip_a/
    - keywords.json
    - ocr.json
    - rag_chunks.json
    - raw_insights.json
    - scenes.json
    - transcript.json
  - clip_b/
    - ...same files...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from videographrag.ingestion.loader import iter_clip_dirs, load_clip_bundle
from videographrag.normalization.normalizer import normalize_clip
from videographrag.clip_builder.builder import build_clip_node


def run_pipeline(root_dir: Path) -> None:
    """Iterate clip folders, run module stages, and aggregate placeholders."""
    for clip_dir in iter_clip_dirs(root_dir):
        raw_clip = load_clip_bundle(clip_dir)
        normalized_clip = normalize_clip(raw_clip)
        _clip_node = build_clip_node(normalized_clip)
        # TODO: semantic -> graph -> retrieval indexing orchestration.


def main() -> None:
    """CLI entrypoint for GraphRAG pipeline bootstrapping."""
    parser = argparse.ArgumentParser(description="Run VideoGraphRAG pipeline.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory containing multiple clip folders.",
    )
    args = parser.parse_args()
    run_pipeline(args.root_dir)


if __name__ == "__main__":
    main()
