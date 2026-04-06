from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import heapq
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..scoring.scorer import cosine_distance


@dataclass
class FrameNode:
    index: int
    token_cost: float
    scores: Dict[str, float]
    frame: Any = None
    entities: List[Any] = field(default_factory=list)
    subtitles: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def score(self, key: str, default: float = 0.0) -> float:
        return float(self.scores.get(key, default))


@dataclass
class ClipCluster:
    id: int
    frames: List[FrameNode]

    @property
    def boundary_frames(self) -> Tuple[FrameNode, FrameNode]:
        return self.frames[0], self.frames[-1]

    @property
    def total_token_cost(self) -> float:
        return calculate_token_cost(self)

    @property
    def cluster_score_profile(self) -> Dict[str, float]:
        if not self.frames:
            return {
                "semantic_avg": 0.0,
                "motion_max": 0.0,
                "entity_avg": 0.0,
                "consistency_avg": 0.0,
                "total_avg": 0.0,
            }
        semantic = [f.score("semantic") for f in self.frames]
        motion = [f.score("motion") for f in self.frames]
        entity = [f.score("entity") for f in self.frames]
        consistency = [f.score("consistency") for f in self.frames]
        total = [f.score("total") for f in self.frames]
        return {
            "semantic_avg": float(np.mean(semantic)),
            "motion_max": float(np.max(motion)),
            "entity_avg": float(np.mean(entity)),
            "consistency_avg": float(np.mean(consistency)),
            "total_avg": float(np.mean(total)),
        }


def _get(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _normalize_subtitles(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if x is not None and str(x).strip()]
    return [str(val)]


def _estimate_text_tokens(texts: Iterable[str]) -> int:
    c = 0
    for t in texts:
        c += len(str(t).split())
    return c


def build_frame_node(
    raw_frame: Any,
    index: int,
    default_visual_token_cost: float = 30.0,
) -> FrameNode:
    scores = _get(raw_frame, "scores", {}) or {}
    if not isinstance(scores, dict):
        scores = {}

    token_cost = _get(raw_frame, "token_cost", None)
    if token_cost is None:
        token_cost = _get(raw_frame, "base_token_cost", default_visual_token_cost)

    entities = _get(raw_frame, "entities", []) or []
    subtitles = (
        _normalize_subtitles(_get(raw_frame, "subtitles", None))
        or _normalize_subtitles(_get(raw_frame, "subtitle", None))
        or _normalize_subtitles(_get(raw_frame, "text", None))
    )
    embedding = _get(raw_frame, "embedding", None)
    if embedding is None:
        embedding = _get(raw_frame, "frame_embedding", None)

    return FrameNode(
        index=index,
        token_cost=float(token_cost),
        scores={k: float(v) for k, v in scores.items()},
        frame=raw_frame,
        entities=list(entities),
        subtitles=subtitles,
        embedding=embedding,
    )


def calculate_merge_affinity(
    cluster_a: ClipCluster,
    cluster_b: ClipCluster,
    w_consistency: float = 0.65,
    w_entity: float = 0.25,
    w_semantic: float = 0.10,
) -> float:
    a_last = cluster_a.boundary_frames[1]
    b_first = cluster_b.boundary_frames[0]

    consistency = 0.5 * (a_last.score("consistency") + b_first.score("consistency"))
    entity_delta = 0.5 * (a_last.score("entity") + b_first.score("entity"))
    sem_a = a_last.score("semantic")
    sem_b = b_first.score("semantic")
    semantic_continuity = max(0.0, 1.0 - abs(sem_a - sem_b))

    affinity = (
        (w_consistency * consistency)
        + (w_entity * (1.0 - entity_delta))
        + (w_semantic * semantic_continuity)
    )
    return float(np.clip(affinity, 0.0, 1.0))


def calculate_token_cost(
    cluster: ClipCluster,
    entity_token_cost: float = 2.0,
    subtitle_token_cost: float = 0.5,
) -> float:
    base = sum(f.token_cost for f in cluster.frames)

    unique_entities: Set[str] = set()
    for f in cluster.frames:
        for e in f.entities:
            if isinstance(e, dict):
                cls = e.get("class_id", e.get("label", str(e)))
                unique_entities.add(str(cls))
            else:
                unique_entities.add(str(e))

    subtitle_words = []
    for f in cluster.frames:
        subtitle_words.extend(f.subtitles)

    dynamic = (len(unique_entities) * entity_token_cost) + (
        _estimate_text_tokens(subtitle_words) * subtitle_token_cost
    )
    return float(base + dynamic)


def _frame_keep_priority(frames: List[FrameNode], idx: int) -> float:
    cur = frames[idx]
    gate = cur.score("gate")
    semantic = cur.score("semantic")
    motion = cur.score("motion")
    total = cur.score("total")

    similarity_bonus = 0.0
    if 0 < idx < len(frames) - 1:
        prev_e = frames[idx - 1].embedding
        next_e = frames[idx + 1].embedding
        cur_e = cur.embedding
        if prev_e is not None and cur_e is not None and next_e is not None:
            sim_prev = 1.0 - cosine_distance(prev_e, cur_e)
            sim_next = 1.0 - cosine_distance(cur_e, next_e)
            similarity_bonus = 0.5 * (sim_prev + sim_next)

    # lower => better candidate to drop
    return float(
        (0.45 * gate)
        + (0.25 * semantic)
        + (0.15 * motion)
        + (0.15 * total)
        - (0.30 * similarity_bonus)
    )


def adaptive_squeeze(
    tentative_cluster: ClipCluster,
    token_limit: float,
    max_drop_ratio: float = 0.5,
    entity_token_cost: float = 2.0,
    subtitle_token_cost: float = 0.5,
) -> Tuple[bool, ClipCluster]:
    frames = list(tentative_cluster.frames)
    n = len(frames)
    if n <= 2:
        return (
            calculate_token_cost(
                tentative_cluster,
                entity_token_cost=entity_token_cost,
                subtitle_token_cost=subtitle_token_cost,
            )
            <= token_limit,
            tentative_cluster,
        )

    max_drop = max(1, int(n * max_drop_ratio))
    dropped = 0

    candidate = ClipCluster(id=tentative_cluster.id, frames=frames)
    while (
        calculate_token_cost(
            candidate,
            entity_token_cost=entity_token_cost,
            subtitle_token_cost=subtitle_token_cost,
        )
        > token_limit
        and dropped < max_drop
        and len(candidate.frames) > 2
    ):
        # lock first and last frame
        internal_idxs = range(1, len(candidate.frames) - 1)
        drop_idx = min(internal_idxs, key=lambda i: _frame_keep_priority(candidate.frames, i))
        candidate.frames.pop(drop_idx)
        dropped += 1

    ok = (
        calculate_token_cost(
            candidate,
            entity_token_cost=entity_token_cost,
            subtitle_token_cost=subtitle_token_cost,
        )
        <= token_limit
    )
    return ok, candidate


def group_frames(
    raw_frames: List[Any],
    effective_context_limit: float,
    base_merge_threshold: float = 0.55,
    default_visual_token_cost: float = 30.0,
    entity_token_cost: float = 2.0,
    subtitle_token_cost: float = 0.5,
    max_drop_ratio: float = 0.5,
) -> List[ClipCluster]:
    if not raw_frames:
        return []

    nodes = [
        build_frame_node(f, i, default_visual_token_cost=default_visual_token_cost)
        for i, f in enumerate(raw_frames)
    ]
    clusters: Dict[int, ClipCluster] = {i: ClipCluster(id=i, frames=[n]) for i, n in enumerate(nodes)}

    prev_id: Dict[int, Optional[int]] = {i: (i - 1 if i > 0 else None) for i in range(len(nodes))}
    next_id: Dict[int, Optional[int]] = {
        i: (i + 1 if i < len(nodes) - 1 else None) for i in range(len(nodes))
    }
    active: Set[int] = set(clusters.keys())

    heap: List[Tuple[float, int, int, int]] = []
    serial = count()
    incompatible: Set[Tuple[int, int]] = set()
    next_cluster_id = len(nodes)

    def _push_pair(left: Optional[int], right: Optional[int]) -> None:
        if left is None or right is None:
            return
        if left not in active or right not in active:
            return
        if next_id.get(left) != right or prev_id.get(right) != left:
            return
        if (left, right) in incompatible:
            return
        affinity = calculate_merge_affinity(clusters[left], clusters[right])
        heapq.heappush(heap, (-affinity, left, right, next(serial)))

    for i in range(len(nodes) - 1):
        _push_pair(i, i + 1)

    while heap:
        neg_aff, left, right, _ = heapq.heappop(heap)
        affinity = -neg_aff
        if affinity < base_merge_threshold:
            break

        if left not in active or right not in active:
            continue
        if next_id.get(left) != right or prev_id.get(right) != left:
            continue
        if (left, right) in incompatible:
            continue

        merged_frames = clusters[left].frames + clusters[right].frames
        tentative = ClipCluster(id=next_cluster_id, frames=merged_frames)
        cost = calculate_token_cost(
            tentative,
            entity_token_cost=entity_token_cost,
            subtitle_token_cost=subtitle_token_cost,
        )

        if cost <= effective_context_limit:
            ok, accepted = True, tentative
        else:
            ok, accepted = adaptive_squeeze(
                tentative,
                token_limit=effective_context_limit,
                max_drop_ratio=max_drop_ratio,
                entity_token_cost=entity_token_cost,
                subtitle_token_cost=subtitle_token_cost,
            )

        if not ok:
            incompatible.add((left, right))
            continue

        accepted.id = next_cluster_id
        clusters[next_cluster_id] = accepted

        l_prev = prev_id.get(left)
        r_next = next_id.get(right)

        active.discard(left)
        active.discard(right)
        active.add(next_cluster_id)

        prev_id[next_cluster_id] = l_prev
        next_id[next_cluster_id] = r_next

        if l_prev is not None:
            next_id[l_prev] = next_cluster_id
        if r_next is not None:
            prev_id[r_next] = next_cluster_id

        prev_id.pop(left, None)
        prev_id.pop(right, None)
        next_id.pop(left, None)
        next_id.pop(right, None)

        _push_pair(l_prev, next_cluster_id)
        _push_pair(next_cluster_id, r_next)

        next_cluster_id += 1

    # collect clusters in chronological order
    if not active:
        return []

    head = None
    for cid in active:
        if prev_id.get(cid) is None:
            head = cid
            break
    if head is None:
        head = min(active)

    out: List[ClipCluster] = []
    cur = head
    seen = set()
    while cur is not None and cur not in seen and cur in active:
        out.append(clusters[cur])
        seen.add(cur)
        cur = next_id.get(cur)

    return out


# PascalCase aliases from the plan
Calculate_Merge_Affinity = calculate_merge_affinity
Calculate_Token_Cost = calculate_token_cost
Adaptive_Squeeze = adaptive_squeeze


__all__ = [
    "FrameNode",
    "ClipCluster",
    "build_frame_node",
    "calculate_merge_affinity",
    "calculate_token_cost",
    "adaptive_squeeze",
    "group_frames",
    "Calculate_Merge_Affinity",
    "Calculate_Token_Cost",
    "Adaptive_Squeeze",
]
