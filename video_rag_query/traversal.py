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

        for step in typed_steps:
            if not states:
                logger.warning("All states pruned before plan completion.")
                break

            logger.info(
                f"Step {getattr(step, 'step', '?')} "
                f"op={getattr(step, 'operation', '?')} "
                f"active_states={len(states)}"
            )

            states = self._dispatch(step, states)
            states = self._prune(states)

        # ── collect & re-rank ────────────────────────────────────────────────
        candidates = self._collect_candidates(states)
        candidates = self._rerank(candidates, decomposition)
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

                # Edge-aware scoring
                normalised_weight = min(edge.weight / 5.0, 1.0)  # cap at 1.0
                s.edge_weight_score = max(s.edge_weight_score, normalised_weight)

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
                if op == "eq" and str(node_val).lower() == str(value).lower():
                    keep = True
                elif op == "contains" and str(value).lower() in str(node_val).lower():
                    keep = True
                elif op == "gt" and float(node_val) > float(value):
                    keep = True
                elif op == "lt" and float(node_val) < float(value):
                    keep = True
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

        This is a terminal-ish step — it enriches the state but does not
        expand or prune.
        """
        target_label = step.target  # "EntityRef" or "ClipRef"
        fields = step.fields

        for state in states:
            # Collect node IDs that match the target label heuristic
            target_ids: List[str] = []
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
                f"Extracted {fields} from {len(extracted)} {target_label} nodes"
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
    ) -> List[TraversalResult]:
        """
        Final re-ranking pass over collected candidates.

        Currently uses:
          • Path score (already computed)
          • Entity relevance boost (resolved entities present in path)
          • Confidence from decomposition

        This is the integration point for future semantic re-ranking
        (query ↔ clip summary embedding similarity).
        """
        if not candidates:
            return []

        resolved_ids = {
            e.resolved_entity_id
            for e in decomposition.entities
            if e.resolved_entity_id
        }

        for c in candidates:
            # Boost if the result path contains a resolved entity
            entity_overlap = len(set(c.entities) & resolved_ids)
            if entity_overlap > 0:
                c.score += 0.05 * entity_overlap

            # Scale by decomposition confidence
            if decomposition.confidence > 0:
                c.score *= (0.5 + 0.5 * decomposition.confidence)

            c.score = round(max(0.0, min(1.0, c.score)), 4)

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates
