"""
mapping_store.py
~~~~~~~~~~~~~~~~
SQLite-backed store for the Entity→Clip bipartite mapping and the
Clip→Clip SHARES_ENTITY similarity index.

Replaces the Neo4j Mapping Graph instance entirely.  No server required —
the store is a single `.db` file that lives alongside the pipeline outputs.

Schema
------
  entity_clip_map(entity_id TEXT, clip_id TEXT, confidence REAL,
                  source TEXT, timestamp REAL)

  clip_similarity(clip1_id TEXT, clip2_id TEXT, weight REAL)

Both tables are indexed for fast lookup.
"""

import sqlite3
import logging
import math
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------

class MappingStore:
    """
    Thin wrapper around a SQLite database that holds the bipartite
    entity↔clip mapping and clip-to-clip similarity edges.

    Usage
    -----
    with MappingStore("outputs/mapping.db") as store:
        store.insert_mappings([...])
        results = store.get_clips_for_entity("person_john_doe")
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def open(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        logger.info("MappingStore opened: %s", self.db_path)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("MappingStore closed: %s", self.db_path)

    # ── Schema ─────────────────────────────────────────────────────────────

    def _create_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entity_clip_map (
                entity_id  TEXT NOT NULL,
                clip_id    TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source     TEXT NOT NULL DEFAULT '',
                timestamp  REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (entity_id, clip_id, source)
            );

            CREATE INDEX IF NOT EXISTS idx_ecm_entity ON entity_clip_map(entity_id);
            CREATE INDEX IF NOT EXISTS idx_ecm_clip   ON entity_clip_map(clip_id);

            CREATE TABLE IF NOT EXISTS clip_similarity (
                clip1_id TEXT NOT NULL,
                clip2_id TEXT NOT NULL,
                weight   REAL NOT NULL,
                PRIMARY KEY (clip1_id, clip2_id)
            );

            CREATE INDEX IF NOT EXISTS idx_cs_clip1 ON clip_similarity(clip1_id);
            CREATE INDEX IF NOT EXISTS idx_cs_clip2 ON clip_similarity(clip2_id);
        """)
        self._conn.commit()

    # ── Write API ──────────────────────────────────────────────────────────

    def insert_mappings(self, mappings: List[Dict[str, Any]], batch_size: int = 5000):
        """
        Bulk-insert entity→clip mappings.

        Each mapping dict must have keys:
            entity_id, clip_id, confidence, source, timestamp
        """
        if not mappings:
            return

        rows = [
            (m["entity_id"], m["clip_id"],
             float(m.get("confidence", 1.0)),
             str(m.get("source", "")),
             float(m.get("timestamp", 0.0)))
            for m in mappings
        ]

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._conn.executemany(
                """INSERT OR REPLACE INTO entity_clip_map
                   (entity_id, clip_id, confidence, source, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                batch,
            )
        self._conn.commit()
        logger.info("MappingStore: inserted %d entity→clip rows.", len(rows))

    def insert_clip_similarities(self, edges: List[Dict[str, Any]], batch_size: int = 5000):
        """
        Bulk-insert clip-to-clip similarity edges (SHARES_ENTITY equivalent).

        Each edge dict: { clip1_id, clip2_id, weight }
        """
        if not edges:
            return

        rows = [(e["c1"], e["c2"], float(e["weight"])) for e in edges]

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._conn.executemany(
                """INSERT OR REPLACE INTO clip_similarity
                   (clip1_id, clip2_id, weight)
                   VALUES (?, ?, ?)""",
                batch,
            )
        self._conn.commit()
        logger.info("MappingStore: inserted %d clip-similarity rows.", len(rows))

    def clear(self):
        """Truncate all tables (useful for re-runs)."""
        self._conn.executescript("""
            DELETE FROM entity_clip_map;
            DELETE FROM clip_similarity;
        """)
        self._conn.commit()
        logger.info("MappingStore: cleared all tables.")

    # ── Read API ───────────────────────────────────────────────────────────

    def get_clips_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return all clip mappings for a given entity_id."""
        cur = self._conn.execute(
            "SELECT clip_id, confidence, source, timestamp "
            "FROM entity_clip_map WHERE entity_id = ?",
            (entity_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_entities_for_clip(self, clip_id: str) -> List[Dict[str, Any]]:
        """Return all entities that appear in a given clip_id."""
        cur = self._conn.execute(
            "SELECT entity_id, confidence, source, timestamp "
            "FROM entity_clip_map WHERE clip_id = ?",
            (clip_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_similar_clips(self, clip_id: str, min_weight: float = 0.0) -> List[Dict[str, Any]]:
        """Return clips similar to clip_id (SHARES_ENTITY equivalent)."""
        cur = self._conn.execute(
            """SELECT clip2_id AS clip_id, weight
               FROM clip_similarity
               WHERE clip1_id = ? AND weight >= ?
               ORDER BY weight DESC""",
            (clip_id, min_weight),
        )
        return [dict(row) for row in cur.fetchall()]

    def load_bipartite_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the full entity→clips mapping as an in-memory dict.
        Format matches the legacy build_bipartite_mapping_dict() output:
            { entity_id: [ {clip_id, confidence, source, timestamp}, ... ] }
        """
        cur = self._conn.execute(
            "SELECT entity_id, clip_id, confidence, source, timestamp "
            "FROM entity_clip_map ORDER BY entity_id"
        )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in cur.fetchall():
            eid = row["entity_id"]
            if eid not in result:
                result[eid] = []
            result[eid].append({
                "clip_id":    row["clip_id"],
                "confidence": row["confidence"],
                "source":     row["source"],
                "timestamp":  row["timestamp"],
            })
        logger.info("MappingStore: loaded bipartite dict with %d entities.", len(result))
        return result

    def load_clip_similarity_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the clip similarity index as an in-memory dict.
        Ensures bidirectional traversal by adding both directions.
        Format: { clip_id: [ {clip_id, weight}, ... ] }
        """
        cur = self._conn.execute(
            "SELECT clip1_id, clip2_id, weight FROM clip_similarity"
        )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in cur.fetchall():
            c1, c2, w = row["clip1_id"], row["clip2_id"], row["weight"]
            
            # Add c1 -> c2
            if c1 not in result: result[c1] = []
            result[c1].append({"clip_id": c2, "weight": w})
            
            # Add c2 -> c1
            if c2 not in result: result[c2] = []
            result[c2].append({"clip_id": c1, "weight": w})
            
        logger.info("MappingStore: loaded bidirectional clip similarity dict with %d clip entries.", len(result))
        return result

    def stats(self) -> Dict[str, int]:
        """Return row counts for monitoring."""
        ecm = self._conn.execute("SELECT COUNT(*) FROM entity_clip_map").fetchone()[0]
        cs  = self._conn.execute("SELECT COUNT(*) FROM clip_similarity").fetchone()[0]
        return {"entity_clip_mappings": ecm, "clip_similarities": cs}
