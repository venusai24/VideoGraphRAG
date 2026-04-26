"""
mapping_builder.py
~~~~~~~~~~~~~~~~~~
Builds the Entity→Clip bipartite mapping and the Clip→Clip similarity index,
storing both in a SQLite file via MappingStore.

This replaces the previous Neo4j Mapping Graph (Instance 3) entirely,
eliminating the APPEARS_IN / SHARES_ENTITY relationships that were pushing
the database over its 400 000-relationship limit.
"""

import logging
import math
from typing import Dict, Any, List

from semantic_graph import (
    get_entity_id, filter_instances, normalize_name, get_normalized_mapping,
    passes_seen_duration, ENTITY_CATEGORIES, _get_summarized_insights,
)
from graph_store.mapping_store import MappingStore

logger = logging.getLogger(__name__)


class MappingBuilder:
    """
    Computes entity→clip overlaps and clip-to-clip TF-IDF similarity,
    then persists both to a SQLite MappingStore.

    Args:
        db_path: Path to the SQLite file (e.g. ``outputs/mapping.db``).
                 The file is created if it does not exist.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    # ── Public API ─────────────────────────────────────────────────────────

    def build(
        self,
        clip_data: Dict[str, Dict[str, Any]],
        clip_intervals: List[Dict[str, Any]],
        *,
        merge_threshold: float = 0.85,
        significance_threshold: float = 0.5,
        clear_existing: bool = False,
    ) -> "MappingStore":
        """
        Parameters
        ----------
        clip_data:
            Raw video payload dict  { folder_name -> payload }.
        clip_intervals:
            List of dicts with keys  {node_id, video_id, start, end}.
            Typically fetched from the Clip Graph in Neo4j.
        merge_threshold:
            Entity normalisation threshold forwarded to get_normalized_mapping.
        significance_threshold:
            Minimum TF-IDF score for a SHARES_ENTITY edge to be stored.
        clear_existing:
            If True, wipe the existing store before writing.

        Returns
        -------
        The open MappingStore so callers can inspect stats immediately.
        (Caller is responsible for closing it, or use as context manager.)
        """
        if not clip_intervals:
            logger.warning("No clip intervals supplied — mapping store will be empty.")

        store = MappingStore(self.db_path)
        store.open()

        if clear_existing:
            store.clear()

        # ── Canonical entity name resolution ────────────────────────────────
        canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

        def get_canonical(entity, e_type):
            raw_id = get_entity_id(entity, e_type)
            return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

        # ── TF-IDF tracking ─────────────────────────────────────────────────
        # clip_entities[clip_id] = { entity_id: best_confidence }
        clip_entities: Dict[str, Dict[str, float]] = {
            c["node_id"]: {} for c in clip_intervals
        }

        all_mappings: List[Dict[str, Any]] = []

        # ── Main loop ───────────────────────────────────────────────────────
        for folder_name, payloads in clip_data.items():
            si  = _get_summarized_insights(payloads)
            ocr = payloads.get("ocr")

            video_id   = folder_name
            rag_chunks = payloads.get("rag_chunks")
            if rag_chunks and isinstance(rag_chunks, list) and rag_chunks:
                video_id = rag_chunks[0].get("video_id", folder_name)

            # Filter clips that belong to this video
            relevant_clips = [c for c in clip_intervals if c["video_id"] == video_id]

            def _extract(entity_node_id: str, instances_array: list, source_name: str, default_conf: float = 1.0):
                """Map entity instances → overlapping clips and record in all_mappings."""
                valid = filter_instances(instances_array, default_conf)
                for inst in valid:
                    i_start = inst.get("startSeconds") or inst.get("start") or inst.get("startTime", "")
                    i_end   = inst.get("endSeconds")   or inst.get("end")   or inst.get("endTime", "")
                    try:
                        i_start = float(i_start)
                        i_end   = float(i_end)
                    except (ValueError, TypeError):
                        continue

                    confidence = float(inst.get("confidence", default_conf))

                    for clip in relevant_clips:
                        if max(i_start, clip["start"]) < min(i_end, clip["end"]):
                            all_mappings.append({
                                "entity_id": entity_node_id,
                                "clip_id":   clip["node_id"],
                                "confidence": confidence,
                                "source":    source_name,
                                "timestamp": i_start,
                            })
                            # Keep highest confidence per entity per clip for TF-IDF
                            prev = clip_entities[clip["node_id"]].get(entity_node_id, 0.0)
                            if confidence > prev:
                                clip_entities[clip["node_id"]][entity_node_id] = confidence

            if si:
                for category, entity_type in ENTITY_CATEGORIES.items():
                    for entity in si.get(category, []):
                        if not isinstance(entity, dict):
                            continue
                        if not passes_seen_duration(entity):
                            continue
                        node_id = get_canonical(entity, entity_type)
                        if node_id:
                            arrays = entity.get("instances", []) + entity.get("appearances", [])
                            _extract(node_id, arrays, source_name=category)

                # NOTE: Sentiments, emotions, audioEffects, framePatterns excluded — not entity nodes.

            if ocr and isinstance(ocr, list):
                for item in ocr:
                    if not isinstance(item, dict): continue
                    text = item.get("text")
                    if not text: continue
                    node_id_str = f"text_{normalize_name(text)}"
                    arrays = item.get("instances", []) + item.get("appearances", [])
                    _extract(node_id_str, arrays, source_name="OCR")

        # ── Persist entity→clip mappings ────────────────────────────────────
        store.insert_mappings(all_mappings)

        # ── Build clip-to-clip TF-IDF similarity ────────────────────────────
        N = len(clip_intervals)
        entity_df: Dict[str, int] = {}
        for ents in clip_entities.values():
            for e in ents:
                entity_df[e] = entity_df.get(e, 0) + 1

        entity_idf = {e: math.log(N / max(1, df)) for e, df in entity_df.items()}

        similarity_edges: List[Dict[str, Any]] = []
        clip_ids = list(clip_entities.keys())

        for i in range(len(clip_ids)):
            c1 = clip_ids[i]
            for j in range(i + 1, len(clip_ids)):
                c2 = clip_ids[j]
                shared = set(clip_entities[c1]).intersection(clip_entities[c2])
                if not shared:
                    continue

                score = sum(
                    ((clip_entities[c1][e] + clip_entities[c2][e]) / 2.0) * entity_idf[e]
                    for e in shared
                )

                if score >= significance_threshold:
                    # Unidirectional (lower index → higher index).
                    # Retrieval queries both directions explicitly.
                    similarity_edges.append({"c1": c1, "c2": c2, "weight": score})

        store.insert_clip_similarities(similarity_edges)

        stats = store.stats()
        logger.info(
            "MappingBuilder complete. entity→clip: %d rows, clip similarities: %d rows. DB: %s",
            stats["entity_clip_mappings"], stats["clip_similarities"], self.db_path,
        )
        return store
