"""
rebuild_entity_graph.py
~~~~~~~~~~~~~~~~~~~~~~~
Rebuilds the Entity Graph (Neo4j Instance 2) and SQLite MappingStore from
the 8 complete output folders.

- Does NOT delete existing OCR text entities (uses MERGE).
- Adds/updates person, brand, location, topic, label, detected_object,
  keyword, face entities with seenDuration >= 1.0s filter.
- Rebuilds RELATED_TO (co-occurrence) and SUBCLASS_OF (topic hierarchy) edges.
- Re-runs MappingBuilder to update the SQLite entity→clip mapping.

Run from: /mnt/MIG_store/Datasets/blending/madhav/VRAG/video_rag_preprocessing/
  python scratch/rebuild_entity_graph.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import VideoDataLoader
from graph_store.connection import MultiGraphManager
from graph_store.builders.entity_builder import EntityGraphBuilder
from graph_store.builders.mapping_builder import MappingBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUTS_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs"
)

# Only the 8 fully complete folders (have raw_insights + scenes + ocr + transcript + rag_chunks)
COMPLETE_FOLDERS = [
    "303cbc17",
    "43a38484",
    "7264ee86",
    "79f019e3",
    "8798066a",
    "ac1682fb",
    "b120483a",
    "f2083c67",
]

MAPPING_DB_PATH = os.path.join(OUTPUTS_BASE, "mapping.db")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_clip_data() -> dict:
    """Load and merge clip_data dicts from all complete folders."""
    combined = {}
    for folder in COMPLETE_FOLDERS:
        folder_path = os.path.join(OUTPUTS_BASE, folder)
        if not os.path.isdir(folder_path):
            logger.warning(f"Folder not found, skipping: {folder_path}")
            continue
        try:
            loader = VideoDataLoader(folder_path)
            data   = loader.load_data()
            combined.update(data)
            logger.info(f"  Loaded {folder} → {list(data.keys())}")
        except Exception as e:
            logger.error(f"  Failed to load {folder}: {e}")
    logger.info(f"Combined clip_data: {len(combined)} video entries")
    return combined


def fetch_clip_intervals(clip_conn) -> list:
    """Fetch all clip intervals from the Clip Graph Neo4j instance."""
    result = clip_conn.execute_query(
        "MATCH (c:Clip) RETURN c.id AS node_id, c.video_id AS video_id, "
        "c.start AS start, c.end AS end"
    )
    intervals = [
        {
            "node_id":  r["node_id"],
            "video_id": r["video_id"],
            "start":    float(r["start"]),
            "end":      float(r["end"]),
        }
        for r in result
        if r["node_id"] and r["start"] is not None and r["end"] is not None
    ]
    logger.info(f"Fetched {len(intervals)} clip intervals from Clip Graph.")
    return intervals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("═" * 60)
    logger.info("  Entity Graph Rebuild")
    logger.info("═" * 60)

    # 1. Load clip data from all complete folders
    logger.info("\n[1] Loading clip data from complete folders...")
    clip_data = load_all_clip_data()
    if not clip_data:
        logger.error("No clip data loaded. Aborting.")
        sys.exit(1)

    with MultiGraphManager() as manager:
        # 2. Build entity graph (MERGE — keeps existing OCR text entities)
        logger.info("\n[2] Building Entity Graph in Neo4j (MERGE, no delete)...")
        # Clear existing relationships to avoid hitting 400k limit
        logger.info("  Clearing existing relationships in Entity Graph...")
        manager.entity_graph.execute_write("MATCH ()-[r]->() DELETE r")
        
        entity_builder = EntityGraphBuilder(manager.entity_graph)
        entity_builder.build_graph(clip_data)
        logger.info("  Entity Graph build complete.")

        # 3. Verify entity graph counts
        logger.info("\n[3] Entity Graph verification:")
        counts = manager.entity_graph.execute_query(
            "MATCH (e:Entity) RETURN e.type AS type, count(*) AS cnt ORDER BY cnt DESC"
        )
        for row in counts:
            logger.info(f"    {row['type']:20s} → {row['cnt']}")

        rel_counts = manager.entity_graph.execute_query(
            "MATCH ()-[r]->() RETURN type(r) AS rtype, count(*) AS cnt"
        )
        for row in rel_counts:
            logger.info(f"    [{row['rtype']}] → {row['cnt']}")

        # 4. Rebuild SQLite MappingStore
        logger.info(f"\n[4] Rebuilding SQLite MappingStore: {MAPPING_DB_PATH}")
        clip_intervals = fetch_clip_intervals(manager.clip_graph)

        if not clip_intervals:
            logger.warning("No clip intervals found in Clip Graph — mapping store will be empty.")
        else:
            mb = MappingBuilder(db_path=MAPPING_DB_PATH)
            store = mb.build(
                clip_data         = clip_data,
                clip_intervals    = clip_intervals,
                merge_threshold   = 0.85,
                significance_threshold = 0.5,
                clear_existing    = True,   # always rebuild mapping fresh
            )
            stats = store.stats()
            logger.info(f"  MappingStore: {stats['entity_clip_mappings']} entity→clip rows, "
                        f"{stats['clip_similarities']} clip-similarity rows.")
            store.close()

    logger.info("\n═" * 60)
    logger.info("  Rebuild complete.")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
