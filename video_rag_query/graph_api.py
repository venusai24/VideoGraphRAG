"""
GraphAPI — Strict Neo4j abstraction layer for the VideoGraphRAG traversal engine.

All Cypher is confined to this module.  The TraversalExecutor and any other
consumer must call *only* the public methods defined here.

Architecture notes
──────────────────
The Neo4j backend is split across three Aura instances:
  • Clip Graph   — Clip nodes, NEXT edges
  • Entity Graph — Entity nodes, RELATED_TO / SUBCLASS_OF / ASSOCIATED_WITH / EXPRESSED edges
  • Mapping Graph — EntityRef↔ClipRef bipartite (APPEARS_IN, SHARES_ENTITY)

GraphAPI unifies access through `MultiGraphManager` while keeping the
caller fully isolated from Cypher.
"""

import logging
import os
import sys
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

# ── Resolve imports for sibling packages ─────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_preprocessing_dir = os.path.join(_project_root, "video_rag_preprocessing")
if _preprocessing_dir not in sys.path:
    sys.path.insert(0, _preprocessing_dir)
if os.path.join(_preprocessing_dir, "config") not in sys.path:
    sys.path.insert(0, os.path.join(_preprocessing_dir, "config"))

from graph_store.connection import MultiGraphManager, GraphConnection  # noqa: E402
from graph_store.mapping_store import MappingStore  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)


# ── Data classes returned by GraphAPI ────────────────────────────────────────

@dataclass
class EntityMatch:
    """A single entity candidate returned by find_entity / vector_search."""
    entity_id: str
    name: str
    entity_type: str
    score: float  # 0.0 – 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraversalEdge:
    """One edge discovered during a traverse / temporal_traverse call."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


# ── GraphAPI ─────────────────────────────────────────────────────────────────

class GraphAPI:
    """
    Strict abstraction over the three Neo4j instances.

    All public methods return plain Python dataclasses — never raw Neo4j
    records or Cypher strings.
    """

    def __init__(self, manager: Optional[MultiGraphManager] = None, mapping_db_path: Optional[str] = None):
        self._mgr = manager or MultiGraphManager()
        self._mapping_db_path = mapping_db_path
        self._mapping_store: Optional[MappingStore] = None
        self._connected = False
        # Caches
        self._entity_cache: Dict[str, List[EntityMatch]] = {}
        self._property_cache: Dict[str, Dict[str, Any]] = {}
        self._traverse_cache: Dict[str, List[TraversalEdge]] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if not self._connected:
            self._mgr.connect_all()
            if self._mapping_db_path:
                self._mapping_store = MappingStore(self._mapping_db_path)
                self._mapping_store.open()
                logger.info(f"GraphAPI: connected to Neo4j and SQLite MappingStore at {self._mapping_db_path}")
            else:
                logger.warning("GraphAPI: No mapping_db_path provided. Mapping-based traversals will be empty.")
            self._connected = True

    def close(self) -> None:
        if self._connected:
            self._mgr.close_all()
            if self._mapping_store:
                self._mapping_store.close()
            self._connected = False
            logger.info("GraphAPI: all connections closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    @property
    def _clip(self) -> GraphConnection:
        return self._mgr.clip_graph

    @property
    def _entity(self) -> GraphConnection:
        return self._mgr.entity_graph

    # ── 1. Entity resolution ────────────────────────────────────────────────

    def find_entity(
        self,
        query: str,
        entity_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[EntityMatch]:
        """
        Hybrid entity resolution:
          1. Exact match on normalised name  (score = 1.0)
          2. Fuzzy / substring match         (score based on Levenshtein ratio)

        Returns up to *top_k* candidates sorted by score descending.
        """
        self._ensure_connected()

        cache_key = f"{query}||{entity_type}||{top_k}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        query_norm = query.strip().lower()

        # --- Step 1: exact match ------------------------------------------------
        if entity_type:
            cypher = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) = $name AND e.type = $etype "
                "RETURN e.id AS id, e.name AS name, e.type AS type, "
                "       e.description AS description"
            )
            params: Dict[str, Any] = {"name": query_norm, "etype": entity_type}
        else:
            cypher = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) = $name "
                "RETURN e.id AS id, e.name AS name, e.type AS type, "
                "       e.description AS description"
            )
            params = {"name": query_norm}

        records = self._entity.execute_query(cypher, params)
        results: List[EntityMatch] = []
        seen_ids: Set[str] = set()

        for r in records:
            eid = r["id"]
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            results.append(EntityMatch(
                entity_id=eid,
                name=r.get("name", ""),
                entity_type=r.get("type", ""),
                score=1.0,
                properties={"description": r.get("description", "")},
            ))

        # --- Step 2: fuzzy / CONTAINS fallback -----------------------------------
        if len(results) < top_k:
            if entity_type:
                cypher_fuzzy = (
                    "MATCH (e:Entity) "
                    "WHERE toLower(e.name) CONTAINS $name AND e.type = $etype "
                    "AND NOT toLower(e.name) = $name "
                    "RETURN e.id AS id, e.name AS name, e.type AS type, "
                    "       e.description AS description "
                    "LIMIT $lim"
                )
                params_fuzzy = {"name": query_norm, "etype": entity_type,
                                "lim": top_k - len(results)}
            else:
                cypher_fuzzy = (
                    "MATCH (e:Entity) "
                    "WHERE toLower(e.name) CONTAINS $name "
                    "AND NOT toLower(e.name) = $name "
                    "RETURN e.id AS id, e.name AS name, e.type AS type, "
                    "       e.description AS description "
                    "LIMIT $lim"
                )
                params_fuzzy = {"name": query_norm,
                                "lim": top_k - len(results)}

            fuzzy_records = self._entity.execute_query(cypher_fuzzy, params_fuzzy)
            for r in fuzzy_records:
                eid = r["id"]
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)
                # Score by length ratio (longer substring match → higher score)
                name_lower = (r.get("name") or "").lower()
                ratio = len(query_norm) / max(len(name_lower), 1)
                score = min(0.95, 0.5 + 0.45 * ratio)  # cap below exact
                results.append(EntityMatch(
                    entity_id=eid,
                    name=r.get("name", ""),
                    entity_type=r.get("type", ""),
                    score=round(score, 4),
                    properties={"description": r.get("description", "")},
                ))

        results.sort(key=lambda m: m.score, reverse=True)
        results = results[:top_k]
        self._entity_cache[cache_key] = results
        return results

    # ── 2. Traversal (APPEARS_IN / RELATED_TO / SHARES_ENTITY) ──────────────

    def traverse(
        self,
        node_ids: List[str],
        edge_types: List[str],
    ) -> List[TraversalEdge]:
        """
        Expand *node_ids* along the specified *edge_types*.

        Dispatches to the correct Neo4j instance based on edge type:
          • APPEARS_IN, SHARES_ENTITY → Mapping graph (EntityRef / ClipRef)
          • RELATED_TO                → Entity graph  (Entity nodes)
        """
        self._ensure_connected()

        cache_key = f"{','.join(sorted(node_ids))}||{','.join(sorted(edge_types))}"
        if cache_key in self._traverse_cache:
            return self._traverse_cache[cache_key]

        results: List[TraversalEdge] = []

        for edge_type in edge_types:
            if edge_type in ("APPEARS_IN", "SHARES_ENTITY"):
                edges = self._traverse_mapping(node_ids, edge_type)
            elif edge_type == "RELATED_TO":
                edges = self._traverse_entity(node_ids, edge_type)
            else:
                logger.warning(f"GraphAPI.traverse: unsupported edge type '{edge_type}'")
                continue
            results.extend(edges)

        self._traverse_cache[cache_key] = results
        return results

    def _traverse_mapping(
        self, node_ids: List[str], edge_type: str
    ) -> List[TraversalEdge]:
        """Traverse APPEARS_IN or SHARES_ENTITY in the SQLite MappingStore."""
        if not self._mapping_store:
            return []

        edges: List[TraversalEdge] = []
        for nid in node_ids:
            if edge_type == "APPEARS_IN":
                # Entity (or EntityRef id) -> Clips
                mappings = self._mapping_store.get_clips_for_entity(nid)
                for m in mappings:
                    edges.append(TraversalEdge(
                        source_id=nid,
                        target_id=m["clip_id"],
                        edge_type="APPEARS_IN",
                        weight=float(m.get("confidence", 1.0)),
                        properties={
                            "source": m.get("source"),
                            "timestamp": m.get("timestamp")
                        }
                    ))
            elif edge_type == "SHARES_ENTITY":
                # Clip -> Similar Clips
                sims = self._mapping_store.get_similar_clips(nid)
                for s in sims:
                    edges.append(TraversalEdge(
                        source_id=nid,
                        target_id=s["clip_id"],
                        edge_type="SHARES_ENTITY",
                        weight=float(s.get("weight", 1.0))
                    ))
        return edges

    def _traverse_entity(
        self, node_ids: List[str], edge_type: str
    ) -> List[TraversalEdge]:
        """Traverse RELATED_TO in the Entity graph."""
        cypher = (
            "UNWIND $ids AS nid "
            "MATCH (a:Entity {id: nid})-[r:" + edge_type + "]->(b:Entity) "
            "RETURN a.id AS source, b.id AS target, type(r) AS etype, "
            "       r.weight AS weight, r.relationship_type AS rel_type"
        )
        records = self._entity.execute_query(cypher, {"ids": node_ids})
        edges: List[TraversalEdge] = []
        for r in records:
            edges.append(TraversalEdge(
                source_id=r["source"],
                target_id=r["target"],
                edge_type=edge_type,
                weight=float(r.get("weight") or 1.0),
                properties={
                    k: v for k, v in r.items()
                    if k not in ("source", "target", "etype") and v is not None
                },
            ))
        return edges

    # ── 3. Temporal traversal (NEXT edges in Clip graph) ────────────────────

    def temporal_traverse(
        self,
        clip_ids: List[str],
        direction: str = "forward",
        k: int = 5,
    ) -> List[TraversalEdge]:
        """
        Walk NEXT edges in the Clip graph.

        direction:
          • "forward"  → follow (source)-[:NEXT]->(target)
          • "backward" → reverse
          • "neutral"  → both directions
        """
        self._ensure_connected()

        results: List[TraversalEdge] = []
        directions = []
        if direction in ("forward", "neutral"):
            directions.append("fwd")
        if direction in ("backward", "neutral"):
            directions.append("bwd")

        for d in directions:
            if d == "fwd":
                cypher = (
                    "UNWIND $ids AS cid "
                    "MATCH (c:Clip {id: cid})-[:NEXT*1.." + str(k) + "]->(n:Clip) "
                    "RETURN c.id AS source, n.id AS target, "
                    "       n.start AS t_start, n.end AS t_end"
                )
            else:
                cypher = (
                    "UNWIND $ids AS cid "
                    "MATCH (c:Clip {id: cid})<-[:NEXT*1.." + str(k) + "]-(n:Clip) "
                    "RETURN c.id AS source, n.id AS target, "
                    "       n.start AS t_start, n.end AS t_end"
                )

            records = self._clip.execute_query(cypher, {"ids": clip_ids})
            for r in records:
                src = r["source"]
                tgt = r["target"]
                # Estimate hop distance from source/target temporal positions
                results.append(TraversalEdge(
                    source_id=src,
                    target_id=tgt,
                    edge_type="NEXT",
                    weight=1.0,
                    properties={
                        "direction": d,
                        "start": r.get("t_start"),
                        "end": r.get("t_end"),
                    },
                ))

        return results

    # ── 4. Neighbour lookup ─────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_ids: List[str],
        edge_type: str,
    ) -> Dict[str, List[str]]:
        """
        Return a mapping  source_id → [neighbour_ids]  for a single edge type.
        Thin wrapper over traverse() for convenience.
        """
        edges = self.traverse(node_ids, [edge_type])
        neighbours: Dict[str, List[str]] = {}
        for e in edges:
            neighbours.setdefault(e.source_id, []).append(e.target_id)
        return neighbours

    # ── 5. Node property fetch ──────────────────────────────────────────────

    def get_node_properties(
        self,
        node_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch properties for a list of node IDs.

        Queries all three graphs and merges results — a node_id will only
        exist in one graph, so there is no collision risk.
        """
        self._ensure_connected()

        # Check cache first
        uncached = [nid for nid in node_ids if nid not in self._property_cache]
        if uncached:
            self._fetch_properties_batch(uncached)

        return {nid: self._property_cache.get(nid, {}) for nid in node_ids}

    def _fetch_properties_batch(self, node_ids: List[str]) -> None:
        """Internal: query all three graphs and populate the property cache."""
        # Clip graph
        cypher_clip = (
            "UNWIND $ids AS nid "
            "MATCH (c:Clip {id: nid}) "
            "RETURN c.id AS id, c.video_id AS video_id, "
            "       c.start AS start, c.end AS end, "
            "       c.transcript AS transcript, c.ocr AS ocr, "
            "       c.keywords AS keywords, c.summary AS summary, "
            "       c.clip_path AS clip_path, "
            "       c.average_sentiment AS average_sentiment, "
            "       c.emotion AS emotion, c.speaker_ids AS speaker_ids"
        )
        for r in self._clip.execute_query(cypher_clip, {"ids": node_ids}):
            self._property_cache[r["id"]] = {
                k: v for k, v in r.items() if v is not None
            }

        # Entity graph
        cypher_entity = (
            "UNWIND $ids AS nid "
            "MATCH (e:Entity {id: nid}) "
            "RETURN e.id AS id, e.type AS type, e.name AS name, "
            "       e.description AS description"
        )
        for r in self._entity.execute_query(cypher_entity, {"ids": node_ids}):
            self._property_cache[r["id"]] = {
                k: v for k, v in r.items() if v is not None
            }

        # Mapping graph shortcuts are no longer needed as Entity/Clip properties
        # are fully covered by the above queries. We just ensure all requested
        # IDs are represented in the cache.
        for nid in node_ids:
            if nid not in self._property_cache:
                self._property_cache[nid] = {}

    # ── 6. Keyword fallback search ──────────────────────────────────────────

    def keyword_fallback_search(
        self,
        keywords: List[str],
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Full-text keyword search across clip transcripts, OCR, summaries, and
        keyword fields.  Used as a last-resort retrieval when entity resolution
        fails to produce any graph candidates.

        Returns a list of dicts with keys: clip_id, score, match_fields.
        Clips are ranked by weighted keyword hit count.

        Weight scheme:
            transcript match → 1.2
            ocr match        → 1.0
            keywords match   → 1.0
            summary match    → 0.5
        """
        self._ensure_connected()

        if not keywords:
            return []

        # Build a Cypher query that checks all text fields for any keyword
        # We use toLower(field) CONTAINS toLower(keyword) for case-insensitive matching
        conditions = []
        for i, kw in enumerate(keywords[:5]):  # Cap at 5 keywords to avoid Cypher explosion
            param_name = f"kw{i}"
            conditions.append(
                f"(CASE WHEN toLower(c.transcript) CONTAINS ${param_name} THEN 1.2 ELSE 0 END + "
                f"CASE WHEN toLower(c.ocr) CONTAINS ${param_name} THEN 1.0 ELSE 0 END + "
                f"CASE WHEN toLower(c.keywords) CONTAINS ${param_name} THEN 1.0 ELSE 0 END + "
                f"CASE WHEN toLower(c.summary) CONTAINS ${param_name} THEN 0.5 ELSE 0 END)"
            )

        score_expr = " + ".join(conditions) if conditions else "0"
        params = {f"kw{i}": kw.lower() for i, kw in enumerate(keywords[:5])}

        cypher = (
            f"MATCH (c:Clip) "
            f"WITH c, ({score_expr}) AS relevance "
            f"WHERE relevance > 0 "
            f"RETURN c.id AS clip_id, c.video_id AS video_id, "
            f"       c.start AS start, c.end AS end, "
            f"       c.transcript AS transcript, c.summary AS summary, "
            f"       relevance "
            f"ORDER BY relevance DESC "
            f"LIMIT $top_k"
        )
        params["top_k"] = top_k

        try:
            records = self._clip.execute_query(cypher, params)
        except Exception as e:
            logger.error(f"keyword_fallback_search failed: {e}")
            return []

        results = []
        for r in records:
            results.append({
                "clip_id": r["clip_id"],
                "video_id": r.get("video_id"),
                "start": r.get("start"),
                "end": r.get("end"),
                "score": float(r.get("relevance", 0)),
                "transcript_snippet": (r.get("transcript") or "")[:200],
                "summary": r.get("summary") or "",
            })

        logger.info(f"keyword_fallback_search: {len(results)} clips for keywords={keywords[:5]}")
        return results

    # ── 7. Cache management ─────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Flush all internal caches."""
        self._entity_cache.clear()
        self._property_cache.clear()
        self._traverse_cache.clear()
        logger.debug("GraphAPI: caches cleared.")
