#!/usr/bin/env python3
"""
Populate Graph 1 Clip nodes with local clip file paths.

Deterministic mapping rule per video_id:
- sort Clip nodes by (start, end, id)
- sort local files by clip index from clip_0000.mp4 .. clip_NNNN.mp4
- map by position (node i -> file i)
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
PREPROC_DIR = ROOT_DIR / "video_rag_preprocessing"
if str(PREPROC_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROC_DIR))

from config.neo4j_settings import GraphSettings
from graph_store.connection import GraphConnection

CLIP_FILE_RE = re.compile(r"^clip_(\d{4})\.mp4$")
UPDATE_BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("populate_graph1_clip_paths")


@dataclass(frozen=True)
class ClipNode:
    clip_id: str
    video_id: str
    start: float
    end: float


@dataclass(frozen=True)
class MappingRow:
    clip_id: str
    clip_path: str


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_clip_nodes(conn: GraphConnection) -> List[ClipNode]:
    query = (
        "MATCH (c:Clip) "
        "RETURN c.id AS id, c.video_id AS video_id, c.start AS start, c.end AS end"
    )
    records = conn.execute_query(query)

    nodes: List[ClipNode] = []
    for row in records:
        clip_id = str(row.get("id") or "").strip()
        video_id = str(row.get("video_id") or "").strip()
        start = row.get("start")
        end = row.get("end")

        if not clip_id or not video_id:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        nodes.append(ClipNode(clip_id=clip_id, video_id=video_id, start=start_f, end=end_f))

    return nodes


def resolve_clip_dir(video_id: str) -> Optional[Path]:
    candidates = [
        ROOT_DIR / "outputs" / video_id / "clips",
        ROOT_DIR / "output" / video_id / "clips",
    ]
    return next((path for path in candidates if path.is_dir()), None)


def list_clip_files(clip_dir: Path) -> Tuple[List[Tuple[int, Path]], List[str]]:
    indexed: List[Tuple[int, Path]] = []
    invalid_names: List[str] = []

    for path in sorted(clip_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        match = CLIP_FILE_RE.match(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        idx = int(match.group(1))
        indexed.append((idx, path.resolve()))

    indexed.sort(key=lambda item: item[0])
    return indexed, invalid_names


def ensure_contiguous(indices: List[int]) -> bool:
    return indices == list(range(len(indices)))


def build_mapping_for_video(video_id: str, nodes: List[ClipNode]) -> Tuple[List[MappingRow], Optional[str], bool]:
    clip_dir = resolve_clip_dir(video_id)
    if clip_dir is None:
        return [], f"missing clip directory for video_id={video_id}", True

    indexed_files, invalid_names = list_clip_files(clip_dir)
    if invalid_names:
        return [], f"invalid mp4 file names in {clip_dir}: {invalid_names[:5]}", False

    if not indexed_files:
        return [], f"no clip_*.mp4 files in {clip_dir}", False

    indices = [idx for idx, _ in indexed_files]
    if not ensure_contiguous(indices):
        return [], f"non-contiguous clip indices for {video_id}: first={indices[:5]} ...", False

    sorted_nodes = sorted(nodes, key=lambda n: (n.start, n.end, n.clip_id))
    if len(sorted_nodes) != len(indexed_files):
        return (
            [],
            (
                f"node/file count mismatch for {video_id}: "
                f"nodes={len(sorted_nodes)}, files={len(indexed_files)}"
            ),
            False,
        )

    rows = [
        MappingRow(clip_id=node.clip_id, clip_path=str(path))
        for node, (_, path) in zip(sorted_nodes, indexed_files)
    ]

    return rows, None, False


def apply_updates(conn: GraphConnection, rows: List[MappingRow]) -> None:
    query = (
        "UNWIND $rows AS row "
        "MATCH (c:Clip {id: row.clip_id}) "
        "SET c.clip_path = row.clip_path"
    )

    for batch in batched(rows, UPDATE_BATCH_SIZE):
        payload = [{"clip_id": r.clip_id, "clip_path": r.clip_path} for r in batch]
        conn.execute_write(query, {"rows": payload})


def verify_updates(conn: GraphConnection, rows: List[MappingRow]) -> List[Dict[str, str]]:
    by_id: Dict[str, str] = {r.clip_id: r.clip_path for r in rows}
    invalid: List[Dict[str, str]] = []

    query = (
        "UNWIND $ids AS cid "
        "MATCH (c:Clip {id: cid}) "
        "RETURN c.id AS id, c.clip_path AS clip_path"
    )

    found_ids = set()
    all_ids = list(by_id.keys())
    for id_batch in batched(all_ids, UPDATE_BATCH_SIZE):
        result = conn.execute_query(query, {"ids": id_batch})
        for row in result:
            cid = str(row.get("id") or "")
            cpath = str(row.get("clip_path") or "")
            found_ids.add(cid)

            if not cpath:
                invalid.append({"clip_id": cid, "reason": "empty clip_path"})
                continue
            if cpath != by_id.get(cid):
                invalid.append({"clip_id": cid, "reason": "clip_path mismatch after write"})
                continue
            if not os.path.isfile(cpath):
                invalid.append({"clip_id": cid, "reason": "file missing on disk"})
                continue
            if not os.access(cpath, os.R_OK):
                invalid.append({"clip_id": cid, "reason": "file not readable"})
                continue

    missing_in_graph = set(all_ids) - found_ids
    invalid.extend(
        {"clip_id": cid, "reason": "node not returned by verification query"}
        for cid in sorted(missing_in_graph)
    )

    return invalid


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate Graph 1 Clip.clip_path from local clips")
    parser.add_argument("--apply", action="store_true", help="Apply writes to Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, do not write")
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Optional video_id filter; pass multiple times to process specific videos",
    )
    parser.add_argument(
        "--max-conflicts-report",
        type=int,
        default=50,
        help="Maximum number of conflicts to include in JSON summary",
    )
    args = parser.parse_args()

    if args.apply and args.dry_run:
        parser.error("Use only one of --apply or --dry-run")

    apply_mode = bool(args.apply)
    mode = "apply" if apply_mode else "dry-run"

    conn = GraphConnection(GraphSettings.get_clip_graph_config(), "Clip")

    try:
        nodes = fetch_clip_nodes(conn)
    except Exception as exc:
        logger.error("Failed to query Graph 1 Clip nodes: %s", exc)
        return 1

    if args.video_id:
        requested = set(args.video_id)
        nodes = [n for n in nodes if n.video_id in requested]

    groups: Dict[str, List[ClipNode]] = defaultdict(list)
    for node in nodes:
        groups[node.video_id].append(node)

    all_rows: List[MappingRow] = []
    conflicts: List[Dict[str, str]] = []
    missing_clip_dirs = 0

    for video_id in sorted(groups.keys()):
        rows, error, is_missing_dir = build_mapping_for_video(video_id, groups[video_id])
        if error is not None:
            if is_missing_dir:
                missing_clip_dirs += 1
            conflicts.append({"video_id": video_id, "reason": error})
            continue
        all_rows.extend(rows)

    if apply_mode and all_rows:
        logger.info("Applying %d clip_path updates...", len(all_rows))
        try:
            apply_updates(conn, all_rows)
        except Exception as exc:
            logger.error("Failed while writing clip_path updates: %s", exc)
            return 1

    invalid_paths: List[Dict[str, str]] = []
    if apply_mode and all_rows:
        logger.info("Verifying updated clip_path values...")
        try:
            invalid_paths = verify_updates(conn, all_rows)
        except Exception as exc:
            logger.error("Failed during verification: %s", exc)
            return 1

    sample_map = [
        {"clip_id": row.clip_id, "clip_path": row.clip_path}
        for row in all_rows[: min(10, len(all_rows))]
    ]

    summary = {
        "mode": mode,
        "total_nodes": len(nodes),
        "videos_total": len(groups),
        "videos_with_valid_mapping": len(
            {
                row.clip_id.rsplit("_", 2)[0]
                for row in all_rows
                if "_" in row.clip_id
            }
        ),
        "planned_or_updated": len(all_rows),
        "missing_clip_dirs": missing_clip_dirs,
        "conflict_count": len(conflicts),
        "invalid_paths": len(invalid_paths),
        "conflicts": conflicts[: args.max_conflicts_report],
        "invalid_details": invalid_paths[: args.max_conflicts_report],
        "sample_mappings": sample_map,
    }

    print(json.dumps(summary, indent=2))

    return 2 if (conflicts or invalid_paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())
