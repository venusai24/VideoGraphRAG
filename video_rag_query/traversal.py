"""
TraversalExecutor — Diversity-aware beam-search reasoning engine for VideoGraphRAG.

Executes a structured ``ExecutionPlan`` (from ``models.py``) over the Neo4j
graph using :class:`GraphAPI`, maintaining bounded state, path tracking,
scoring, and re-ranking.

Design highlights
─────────────────
• **Beam search**: keeps at most ``beam_width`` active states after every
  operation, preventing combinatorial explosion.
• **Diversity penalty**: paths that share too many nodes get penalised so
  the result set covers different graph regions.
• **Edge-aware scoring**: each relationship type contributes a tuned boost /
  decay factor.
• **Cycle control**: every ``TraversalState`` tracks ``visited_nodes`` and
  refuses to revisit.
• **Path tracking**: full ordered list of ``(node, edge_type)`` tuples for
  every result, enabling explainability.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from video_rag_query.utils import extract_keywords

from .graph_api import GraphAPI, EntityMatch, TraversalEdge
from .models import (
    ExecutionStep,
    QueryDecomposition,
    StepExtract,
    StepFilter,
    StepResolveEntity,
    StepTemporalTraverse,
    StepTraverse,
)

VALID_OPERATIONS = {"resolve_entity", "traverse", "filter", "temporal_traverse", "extract"}
VALID_EDGES = {"APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"}

# ── Hard traversal limits ──────────────────────────────────────────────────────
MAX_NODES_PER_STEP = 50
MAX_TOTAL_EXPANSIONS = 300
EARLY_STOP_SCORE_THRESHOLD = 0.3
EARLY_STOP_AFTER_STEP = 2

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TraversalConfig:
    """Tunable knobs for the traversal engine."""

    beam_width: int = 15
    """Max active states retained after each operation."""

    min_score_threshold: float = 0.20
    """States with score below this are pruned."""

    max_depth: int = 6
    """Hard limit on traversal hops."""

    temporal_decay_alpha: float = 0.15
    """Exponential decay coefficient for NEXT hops."""

    diversity_penalty_weight: float = 0.20
    """Weight of the Jaccard-based diversity penalty."""

    rerank_top_n: int = 50
    """Number of candidates collected before final re-ranking."""

    # ── scoring weights (must sum to 1.0) ───────────────────────────────────
    w_entity_match: float = 0.25
    w_edge_weight: float = 0.20
    w_temporal: float = 0.20
    w_path_length: float = 0.15
    w_node_confidence: float = 0.10
    w_diversity: float = 0.10


# ── TraversalState ───────────────────────────────────────────────────────────

@dataclass
class TraversalState:
    """A single active hypothesis during beam-search execution."""

    variables: Dict[str, List[str]] = field(default_factory=dict)
    """Mapping from plan variable names to current node IDs."""

    path: List[Tuple[str, str]] = field(default_factory=list)
    """Ordered traversal history: [(node_id, edge_type), ...]."""

    score: float = 1.0
    """Cumulative confidence/quality score."""

    depth: int = 0
    """Number of hops taken so far."""

    visited_nodes: Set[str] = field(default_factory=set)
    """Set of node IDs already visited — used for cycle detection."""

    explanations: List[str] = field(default_factory=list)
    """Human-readable explanations for each operation applied."""

    # ── scoring component accumulators ───────────────────────────────────────
    entity_match_score: float = 0.0
    edge_weight_score: float = 0.0
    temporal_score: float = 1.0
    node_confidence_score: float = 0.0

    def clone(self) -> "TraversalState":
        return TraversalState(
            variables={k: list(v) for k, v in self.variables.items()},
            path=list(self.path),
            score=self.score,
            depth=self.depth,
            visited_nodes=set(self.visited_nodes),
            explanations=list(self.explanations),
            entity_match_score=self.entity_match_score,
            edge_weight_score=self.edge_weight_score,
            temporal_score=self.temporal_score,
            node_confidence_score=self.node_confidence_score,
        )


# ── Traversal result ─────────────────────────────────────────────────────────

@dataclass
class TraversalResult:
    """A single ranked result returned by the executor."""

    clip_id: str
    score: float
    path: List[Tuple[str, str]]
    entities: List[str]
    explanation: str
    best_clip_id: Optional[str] = None  # The specific clip that provided the best keyword match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "score": round(self.score, 4),
            "path": self.path,
            "entities": self.entities,
            "explanation": self.explanation,
        }


# ── TraversalExecutor ────────────────────────────────────────────────────────

class TraversalExecutor:
    """
    State-based beam-search engine that executes a structured ``ExecutionPlan``
    over the Neo4j graph.
    """

    def __init__(
        self,
        api: GraphAPI,
        config: Optional[TraversalConfig] = None,
    ):
        self.api = api
        self.cfg = config or TraversalConfig()

    # ── public API ───────────────────────────────────────────────────────────

    def execute(
        self,
        decomposition: QueryDecomposition,
        original_query: Optional[str] = None,
    ) -> List[TraversalResult]:
        """
        Run the full traversal pipeline:

        1. Parse the execution plan into typed steps.
        2. Initialise states.
        3. For each step: expand → score → prune (beam search).
        4. Collect candidates and re-rank.
        5. Return top-k ``TraversalResult`` objects.
        """
        typed_steps = decomposition.get_typed_execution_plan()
        if not typed_steps:
            logger.warning("Empty execution plan — nothing to execute.")
            return []

        # Seed with a single empty state
        states: List[TraversalState] = [TraversalState()]
        total_expansions = 0
        traversal_telemetry = {
            "nodes_expanded": 0,
            "depth_reached": 0,
            "fallback_triggered": False,
            "early_stop": False,
        }

        for step_idx, step in enumerate(typed_steps):
            if not states:
                logger.warning("All states pruned before plan completion.")
                break

            logger.info(
                f"Step {getattr(step, 'step', '?')} "
                f"op={getattr(step, 'operation', '?')} "
                f"active_states={len(states)}"
            )

            # HARD LIMIT: total expansion budget
            if total_expansions >= MAX_TOTAL_EXPANSIONS:
                logger.warning(
                    f"Traversal expansion limit reached ({MAX_TOTAL_EXPANSIONS}). "
                    f"Stopping early at step {step_idx+1}."
                )
                traversal_telemetry["early_stop"] = True
                break

            states = self._dispatch(step, states)

            # Enforce per-step node limit
            if len(states) > MAX_NODES_PER_STEP:
                states.sort(key=lambda s: s.score, reverse=True)
                states = states[:MAX_NODES_PER_STEP]
                logger.info(f"Per-step limit: pruned to {MAX_NODES_PER_STEP} states")

            total_expansions += len(states)
            states = self._prune(states)

            # Track telemetry
            max_depth = max((s.depth for s in states), default=0)
            traversal_telemetry["depth_reached"] = max(traversal_telemetry["depth_reached"], max_depth)
            traversal_telemetry["nodes_expanded"] = total_expansions

            # Early termination: if avg score < threshold after step 2
            if step_idx >= EARLY_STOP_AFTER_STEP and states:
                avg_score = sum(s.score for s in states) / len(states)
                if avg_score < EARLY_STOP_SCORE_THRESHOLD:
                    logger.warning(
                        f"Early termination: avg score {avg_score:.3f} < {EARLY_STOP_SCORE_THRESHOLD} "
                        f"after step {step_idx+1}. Falling back to keyword search."
                    )
                    traversal_telemetry["early_stop"] = True
                    traversal_telemetry["fallback_triggered"] = True
                    break

        logger.info(f"Traversal telemetry: {traversal_telemetry}")

        # ── collect & re-rank ────────────────────────────────────────────────
        candidates = self._collect_candidates(states)
        candidates = self._rerank(candidates, decomposition, original_query)
        return candidates

    # ── step dispatch ────────────────────────────────────────────────────────

    def _dispatch(
        self, step: ExecutionStep, states: List[TraversalState]
    ) -> List[TraversalState]:
        if isinstance(step, StepResolveEntity):
            return self._op_resolve_entity(step, states)
        if isinstance(step, StepTraverse):
            return self._op_traverse(step, states)
        if isinstance(step, StepTemporalTraverse):
            return self._op_temporal_traverse(step, states)
        if isinstance(step, StepFilter):
            return self._op_filter(step, states)
        if isinstance(step, StepExtract):
            return self._op_extract(step, states)
        logger.warning(f"Unknown step type {type(step).__name__} — skipping.")
        return states

    # ── operations ───────────────────────────────────────────────────────────

    def _op_resolve_entity(
        self, step: StepResolveEntity, states: List[TraversalState]
    ) -> List[TraversalState]:
        """
        Resolve an entity name to graph node IDs.

        Branches the state set: each candidate entity creates a new state
        fork (up to top-k from GraphAPI).
        """
        matches = self.api.find_entity(step.input)
        if not matches:
            logger.warning(f"resolve_entity: no matches for '{step.input}'")
            return []

        new_states: List[TraversalState] = []
        for state in states:
            for match in matches:
                s = state.clone()
                s.variables[step.output] = [match.entity_id]
                s.path.append((match.entity_id, "resolve_entity"))
                s.visited_nodes.add(match.entity_id)
                s.entity_match_score = match.score
                s.score *= match.score
                s.explanations.append(
                    f"Resolved '{step.input}' → {match.entity_id} "
                    f"(score={match.score:.2f})"
                )
                new_states.append(s)

        return new_states

    def _op_traverse(
        self, step: StepTraverse, states: List[TraversalState]
    ) -> List[TraversalState]:
        """
        Expand states along a specified edge type.

        Uses edge-aware boosting:
          • APPEARS_IN  → grounding boost  (×1.2)
          • SHARES_ENTITY → moderate boost  (×1.0, weight-scaled)
          • RELATED_TO → low boost          (×0.8)
        """
        edge_type = step.edge
        from_var = step.from_node
        to_var = step.to_node

        EDGE_BOOST = {
            "APPEARS_IN": 1.2,
            "SHARES_ENTITY": 1.0,
            "RELATED_TO": 0.8,
        }
        boost = EDGE_BOOST.get(edge_type, 1.0)

        new_states: List[TraversalState] = []
        for state in states:
            source_ids = state.variables.get(from_var, [])
            if not source_ids:
                continue

            edges = self.api.traverse(source_ids, [edge_type])
            if not edges:
                continue

            for edge in edges:
                if edge.target_id in state.visited_nodes:
                    continue  # cycle prevention

                if state.depth >= self.cfg.max_depth:
                    continue  # depth limit

                s = state.clone()
                s.variables.setdefault(to_var, [])
                if edge.target_id not in s.variables[to_var]:
                    s.variables[to_var].append(edge.target_id)

                s.path.append((edge.target_id, edge_type))
                s.visited_nodes.add(edge.target_id)
                s.depth += 1

                # Edge-aware scoring: prioritize APPEARS_IN (bipartite)
                normalised_weight = min(edge.weight / 5.0, 1.0)
                if edge_type == "APPEARS_IN":
                    # Confidence from MappingStore is typically 0.0-1.0
                    boost = 1.2 * edge.weight
                else:
                    boost = 1.0

                # Path-length decay
                length_decay = 0.9 ** s.depth
                s.score *= boost * (0.5 + 0.5 * normalised_weight) * length_decay

                s.explanations.append(
                    f"Traversed -{edge_type}-> {edge.target_id} "
                    f"(w={edge.weight:.2f}, boost={boost})"
                )
                new_states.append(s)

        return new_states if new_states else states

    def _op_temporal_traverse(
        self, step: StepTemporalTraverse, states: List[TraversalState]
    ) -> List[TraversalState]:
        """
        Walk NEXT edges from current clip IDs.

        Applies exponential temporal decay: score *= exp(-α * hop_distance).
        """
        direction = step.direction
        limit = min(step.limit or 5, 5)  # hard-cap at 5

        new_states: List[TraversalState] = []
        for state in states:
            # Collect all clip IDs currently in variables
            clip_ids: List[str] = []
            for var_name, ids in state.variables.items():
                for nid in ids:
                    # Heuristic: ClipRef IDs contain underscores with timestamps
                    if "_" in nid and any(c.isdigit() for c in nid):
                        clip_ids.append(nid)

            if not clip_ids:
                new_states.append(state)
                continue

            edges = self.api.temporal_traverse(clip_ids, direction, limit)
            if not edges:
                new_states.append(state)
                continue

            for edge in edges:
                if edge.target_id in state.visited_nodes:
                    continue

                if state.depth >= self.cfg.max_depth:
                    continue

                s = state.clone()

                # Estimate hop distance (rough: based on how many temporal
                # edges separate the two clips — we use depth as proxy)
                hop_distance = 1  # each temporal edge = 1 hop
                temporal_decay = math.exp(
                    -self.cfg.temporal_decay_alpha * hop_distance
                )

                s.variables.setdefault("temporal_clips", [])
                if edge.target_id not in s.variables["temporal_clips"]:
                    s.variables["temporal_clips"].append(edge.target_id)

                s.path.append((edge.target_id, "NEXT"))
                s.visited_nodes.add(edge.target_id)
                s.depth += 1
                s.temporal_score *= temporal_decay
                s.score *= temporal_decay

                s.explanations.append(
                    f"Temporal {direction} → {edge.target_id} "
                    f"(decay={temporal_decay:.3f})"
                )
                new_states.append(s)

        return new_states if new_states else states

    def _op_filter(
        self, step: StepFilter, states: List[TraversalState]
    ) -> List[TraversalState]:
        """
        Drop states whose nodes do not satisfy the filter condition.

        Condition format: {"field": ..., "op": "eq"|"contains"|"gt"|"lt", "value": ...}
        """
        cond = step.condition
        field_name = cond.get("field", "")
        op = cond.get("op", "eq")
        value = cond.get("value", "")

        new_states: List[TraversalState] = []
        for state in states:
            # Gather all node IDs in this state
            all_ids: List[str] = []
            for ids in state.variables.values():
                all_ids.extend(ids)

            if not all_ids:
                continue

            props = self.api.get_node_properties(all_ids)

            # Keep state if ANY node satisfies the condition
            keep = False
            for nid, p in props.items():
                node_val = p.get(field_name)
                if node_val is None:
                    continue
                
                try:
                    if op == "eq":
                        if str(node_val).lower() == str(value).lower():
                            keep = True
                    elif op == "contains":
                        if str(value).lower() in str(node_val).lower():
                            keep = True
                    elif op == "gt":
                        if float(node_val) > float(value):
                            keep = True
                    elif op == "lt":
                        if float(node_val) < float(value):
                            keep = True
                except (ValueError, TypeError):
                    continue
                
                if keep:
                    break

            if keep:
                state.explanations.append(
                    f"Filter passed: {field_name} {op} '{value}'"
                )
                new_states.append(state)

        return new_states

    def _op_extract(
        self, step: StepExtract, states: List[TraversalState]
    ) -> List[TraversalState]:
        """
        Fetch requested fields from target nodes and attach them to the state.
        If target matches a variable name, extract from those nodes only.
        Otherwise, extract from all nodes in all variables.
        """
        target = step.target
        fields = step.fields

        for state in states:
            target_ids: List[str] = []
            if target in state.variables:
                # Targeted extraction from specific variable
                target_ids = state.variables[target]
            else:
                # Fallback: collect all node IDs
                for ids in state.variables.values():
                    target_ids.extend(ids)

            if not target_ids:
                continue

            props = self.api.get_node_properties(target_ids)
            extracted = {}
            for nid, p in props.items():
                extracted[nid] = {f: p.get(f) for f in fields if f in p}

            state.variables["_extracted"] = list(extracted.keys())
            state.explanations.append(
                f"Extracted {fields} from {len(extracted)} {target} nodes"
            )

        return states

    # ── pruning (beam search + early stop) ───────────────────────────────────

    def _prune(self, states: List[TraversalState]) -> List[TraversalState]:
        """
        1. Drop states below ``min_score_threshold``.
        2. Drop states with empty variable mappings.
        3. Apply diversity-aware beam truncation.
        """
        # Early pruning
        alive = [
            s for s in states
            if s.score >= self.cfg.min_score_threshold
            and any(s.variables.values())
        ]

        if len(alive) <= self.cfg.beam_width:
            return alive

        # Diversity-aware beam: penalise similar paths before sorting
        alive = self._apply_diversity_penalty(alive)
        alive.sort(key=lambda s: s.score, reverse=True)
        return alive[: self.cfg.beam_width]

    def _apply_diversity_penalty(
        self, states: List[TraversalState]
    ) -> List[TraversalState]:
        """
        Penalise states whose paths overlap heavily with higher-scored states.

        Uses Jaccard similarity on visited_nodes to compute the penalty.
        """
        # Sort by raw score first so higher-scored states are the "reference"
        states.sort(key=lambda s: s.score, reverse=True)

        for i in range(1, len(states)):
            max_sim = 0.0
            for j in range(i):
                intersection = len(states[i].visited_nodes & states[j].visited_nodes)
                union = len(states[i].visited_nodes | states[j].visited_nodes)
                if union > 0:
                    sim = intersection / union
                    max_sim = max(max_sim, sim)

            penalty = max_sim * self.cfg.diversity_penalty_weight
            states[i].score -= penalty

        return states

    # ── candidate collection ─────────────────────────────────────────────────

    def _collect_candidates(
        self, states: List[TraversalState]
    ) -> List[TraversalResult]:
        """
        Convert surviving states into ``TraversalResult`` objects.

        Identifies clip IDs from variables / paths and deduplicates.
        """
        seen: Set[str] = set()
        results: List[TraversalResult] = []

        for state in states:
            # Find clip IDs — heuristic: IDs with numeric segments
            clip_ids = set()
            entity_ids = set()
            for var, ids in state.variables.items():
                if var.startswith("_"):
                    continue
                for nid in ids:
                    if any(c.isdigit() for c in nid) and "_" in nid:
                        clip_ids.add(nid)
                    else:
                        entity_ids.add(nid)

            # Also scan path
            for node_id, edge_type in state.path:
                if any(c.isdigit() for c in node_id) and "_" in node_id:
                    clip_ids.add(node_id)
                elif edge_type == "resolve_entity":
                    entity_ids.add(node_id)

            if not clip_ids:
                # Fall back: treat all terminal IDs as results
                for ids in state.variables.values():
                    clip_ids.update(ids)

            # Compute final composite score
            final_score = self._composite_score(state)

            for cid in clip_ids:
                if cid in seen:
                    continue
                seen.add(cid)

                results.append(TraversalResult(
                    clip_id=cid,
                    score=final_score,
                    path=state.path,
                    entities=list(entity_ids),
                    explanation=" → ".join(state.explanations),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self.cfg.rerank_top_n]

    # ── composite scoring ────────────────────────────────────────────────────

    def _composite_score(self, state: TraversalState) -> float:
        """
        Weighted combination of the five scoring dimensions plus a diversity
        bonus (encoded as the inverse of depth — shallower = more diverse).

        score = Σ(w_i × component_i)
        """
        path_length_penalty = 0.9 ** state.depth
        diversity_bonus = 1.0 / max(1, len(state.visited_nodes) - 1)

        score = (
            self.cfg.w_entity_match * state.entity_match_score
            + self.cfg.w_edge_weight * state.edge_weight_score
            + self.cfg.w_temporal * state.temporal_score
            + self.cfg.w_path_length * path_length_penalty
            + self.cfg.w_node_confidence * state.node_confidence_score
            + self.cfg.w_diversity * diversity_bonus
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ── re-ranking ───────────────────────────────────────────────────────────

    def _rerank(
        self,
        candidates: List[TraversalResult],
        decomposition: QueryDecomposition,
        original_query: Optional[str] = None,
    ) -> List[TraversalResult]:
        """
        Final re-ranking pass over collected candidates.

        Currently uses:
          • Path score (already computed)
          • Entity relevance boost
          • Confidence from decomposition
          • NEW: Keyword-based overlap boost (lexical matching)
        """
        if not candidates:
            return []

        # 1. Prepare keywords from original query
        query_keywords = set()
        if original_query:
            query_keywords = extract_keywords(original_query)
            logger.info(f"Re-ranking using {len(query_keywords)} query keywords: {query_keywords}")

        resolved_ids = {
            e.resolved_entity_id
            for e in decomposition.entities
            if e.resolved_entity_id
        }

        # 2. Fetch all properties for batch processing
        all_candidate_ids = [c.clip_id for c in candidates]
        all_props = self.api.get_node_properties(all_candidate_ids)

        for c in candidates:
            # A. Entity relevance boost
            entity_overlap = len(set(c.entities) & resolved_ids)
            if entity_overlap > 0:
                c.score += 0.05 * entity_overlap

            # B. Keyword matching boost (Lexical)
            if query_keywords:
                props = all_props.get(c.clip_id, {})
                transcript = str(props.get('transcript', '')).lower()
                ocr = str(props.get('ocr', '')).lower()
                summary = str(props.get('summary', '')).lower()
                keywords_field = str(props.get('keywords', '')).lower()
                
                # If this is an entity, deep-scan its associated clips
                if 'video_id' not in props and self.api._mapping_store:
                    associated_clips = self.api._mapping_store.get_clips_for_entity(c.clip_id)
                    if associated_clips:
                        # Fetch transcripts for associated clips to find the best match
                        clip_ids = [ac['clip_id'] for ac in associated_clips[:3]] # check top 3
                        clip_props = self.api.get_node_properties(clip_ids)
                        
                        best_weighted = 0.0
                        best_cid = None
                        for cid, cp in clip_props.items():
                            c_trans = str(cp.get('transcript', '')).lower()
                            c_ocr = str(cp.get('ocr', '')).lower()
                            
                            c_weighted = 0.0
                            for kw in query_keywords:
                                if kw in c_trans: c_weighted += 1.2
                                elif kw in c_ocr: c_weighted += 1.0
                            
                            if c_weighted > best_weighted:
                                best_weighted = c_weighted
                                best_cid = cid
                        
                        # Use the best associated clip's score if it's better than the entity's own properties
                        if best_cid:
                            transcript = str(clip_props.get(best_cid, {}).get('transcript', ''))
                            ocr = str(clip_props.get(best_cid, {}).get('ocr', ''))
                            c.best_clip_id = best_cid
                
                matches = 0
                weighted_matches = 0.0
                matched_words = []
                for kw in query_keywords:
                    found_in_clip = False
                    if kw in transcript:
                        weighted_matches += 1.2  # Transcript match
                        found_in_clip = True
                    elif kw in ocr:
                        weighted_matches += 1.0  # OCR match
                        found_in_clip = True
                    elif kw in keywords_field:
                        weighted_matches += 1.0  # Keyword match
                        found_in_clip = True
                    elif kw in summary:
                        weighted_matches += 0.5  # Summary match
                        found_in_clip = True
                    
                    if found_in_clip:
                        matches += 1
                        matched_words.append(kw)
            
                # Normalized keyword score (max possible per keyword is 1.2)
                keyword_score = (weighted_matches / (len(query_keywords) * 1.2)) if query_keywords else 0
                
                # Reweight: 30% graph score, 70% deep keyword matching
                c.score = (c.score * 0.3) + (keyword_score * 0.7)
                
                if matches > 0:
                    words_str = ", ".join(matched_words)
                    c.explanation += f" | Deep Match: {matches}/{len(query_keywords)} (w={weighted_matches:.1f}) [{words_str}]"

            # C. Scale by decomposition confidence
            if decomposition.confidence > 0:
                c.score *= (0.5 + 0.5 * decomposition.confidence)

            c.score = round(max(0.0, min(1.0, c.score)), 4)

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates
