"""
Retrieval pipeline for the two-layer VideoGraphRAG NetworkX graph.

Optimized for CPU execution:
  - Sentence-Transformer embedding with all-MiniLM-L6-v2
  - NumPy-vectorized cosine similarity for seed entity selection
  - BM25 ranking via rank_bm25 for final clip scoring
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedClip:
    """A single clip returned by the retrieval pipeline."""
    node_id: str
    video_id: str
    start: float
    end: float
    bm25_score: float
    entity_confidence: float
    semantic_score: float
    final_score: float
    transcript: str = ""
    ocr: str = ""
    summary: str = ""


@dataclass
class RetrievalResult:
    """Full output of a retrieval call."""
    query: str
    seed_entities: List[Tuple[str, float]]   # (node_id, similarity)
    candidate_clip_count: int
    clips: List[RetrievedClip] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokeniser (lightweight, no external deps)
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^a-z0-9]+")

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with", "which", "who", "whom", "what", "where", "when", "how", "why", "has", "been", "somehow", "them"}

def _tokenize(text: str) -> List[str]:
    """Lower-case split with non-alphanumeric separator. Fast on CPU. Filters stopwords. Basic stemming."""
    tokens = []
    for tok in _SPLIT_RE.split(text.lower()):
        if tok and tok not in STOPWORDS:
            # Basic plural removal (e.g. presidents -> president, countries -> countrie)
            if tok.endswith('s') and len(tok) > 3 and not tok.endswith('ss'):
                tok = tok[:-1]
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class VideoGraphRetriever:
    """
    CPU-optimized retrieval over a two-layer NetworkX graph.

    Layer 1 – Clip nodes  (have 'start', 'end', 'video_id', 'transcript', 'ocr')
    Layer 2 – Semantic nodes (have 'type' in {person, brand, location, topic, text})

    Workflow:
      1. Embed the user query with all-MiniLM-L6-v2.
      2. Cosine-similarity against pre-computed entity embeddings → top-k seeds.
      3. Traverse bipartite edges to collect candidate clip nodes.
      4. BM25-rank candidates on concatenated transcript + OCR text.
      5. Return top-n clips.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        bipartite_dict: Dict[str, List[Dict[str, Any]]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        mapping_db_path: Optional[str] = None,
    ):
        self.graph = graph
        self.batch_size = batch_size

        # ── Load mapping from SQLite if path provided ─────────────────────
        if mapping_db_path:
            try:
                from graph_store.mapping_store import MappingStore
                with MappingStore(mapping_db_path) as store:
                    self.bipartite_dict = store.load_bipartite_dict()
                    self._clip_similarity_dict = store.load_clip_similarity_dict()
                logger.info(
                    "Loaded mapping from SQLite: %d entities, %d clip-similarity entries.",
                    len(self.bipartite_dict), len(self._clip_similarity_dict)
                )
            except Exception as e:
                logger.error("Failed to load MappingStore from %s: %s", mapping_db_path, e)
                self.bipartite_dict = bipartite_dict or {}
                self._clip_similarity_dict = {}
        else:
            self.bipartite_dict = bipartite_dict or {}
            self._clip_similarity_dict = {}

        # ── Partition nodes by layer ──────────────────────────────────
        self._entity_ids: List[str] = []
        self._entity_texts: List[str] = []
        self._clip_ids: set = set()
        
        self._known_emotions: set = set()
        self._known_speakers: set = set()

        for node_id, attr in graph.nodes(data=True):
            if attr.get("type") in {"person", "brand", "location", "topic", "text"}:
                # Build a search-friendly text from node name + description
                name = attr.get("name", "") or ""
                desc = attr.get("description", "") or ""
                self._entity_ids.append(node_id)
                
                text_to_embed = f"{name} {desc}".strip()
                if not text_to_embed:
                    text_to_embed = node_id
                self._entity_texts.append(text_to_embed)
            elif "start" in attr and "end" in attr:
                self._clip_ids.add(node_id)
                if attr.get("emotion"):
                    self._known_emotions.add(str(attr["emotion"]).lower())
                for s_id in attr.get("speaker_ids", []):
                    self._known_speakers.add(str(s_id).lower())

        logger.info(
            "Indexed %d entity nodes and %d clip nodes.",
            len(self._entity_ids),
            len(self._clip_ids),
        )

        # ── Load model & pre-compute entity embeddings (one-time cost) ──
        logger.info("Loading SentenceTransformer model: %s ...", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")

        if self._entity_texts:
            logger.info("Pre-computing embeddings for %d entities ...", len(self._entity_texts))
            self._entity_embeddings = self._model.encode(
                self._entity_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # so dot-product == cosine similarity
            )
        else:
            self._entity_embeddings = np.empty((0, 384), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_entities: int = 3,
        top_n_clips: int = 5,
        max_hops: int = 2,
        semantic_decay: float = 0.6,
        temporal_decay: float = 0.9,
        semantic_verification_threshold: float = 0.2
    ) -> RetrievalResult:
        """
        End-to-end retrieval: query → seed entities → candidate clips → BM25 ranking.

        Args:
            query: Natural-language search query.
            top_k_entities: Number of seed entities to select (default 3).
            top_n_clips: Number of final clips to return (default 5).
            max_hops: Depth of clip-to-clip traversal.
            semantic_decay: Decay factor for SHARES_ENTITY edges.
            temporal_decay: Decay factor for NEXT edges.
            semantic_verification_threshold: Minimum semantic score to pass verification.

        Returns:
            A RetrievalResult dataclass with ranked clips.
        """
        if not query or not query.strip():
            return RetrievalResult(query=query, seed_entities=[], candidate_clip_count=0)

        # Step 1 — Embed query (single forward pass)
        query_vec = self._model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]  # shape (384,)

        # Step 2 — Cosine similarity against all entity embeddings (vectorised)
        seed_entities = self._find_seed_entities(query_vec, top_k_entities)

        if not seed_entities:
            return RetrievalResult(
                query=query, seed_entities=[], candidate_clip_count=0
            )

        # Step 3 — Graph traversal: seed entities → clip nodes
        candidate_clips = self._traverse_to_clips(
            seed_entities, max_hops=max_hops, 
            semantic_decay=semantic_decay, temporal_decay=temporal_decay
        )

        if not candidate_clips:
            return RetrievalResult(
                query=query,
                seed_entities=seed_entities,
                candidate_clip_count=0,
            )

        # Step 3.5 — Attribute filtering
        query_lower = query.lower()
        mentioned_emotions = {e for e in self._known_emotions if e in query_lower}
        mentioned_speakers = {s for s in self._known_speakers if f"speaker {s}" in query_lower}
        
        filtered_candidates = {}
        for cid, conf in candidate_clips.items():
            attr = self.graph.nodes[cid]
            keep = True
            if mentioned_emotions:
                if str(attr.get("emotion", "")).lower() not in mentioned_emotions:
                    keep = False
            if mentioned_speakers:
                clip_speakers = [str(s).lower() for s in attr.get("speaker_ids", [])]
                if not any(s in mentioned_speakers for s in clip_speakers):
                    keep = False
            if keep:
                filtered_candidates[cid] = conf

        if not filtered_candidates:
            return RetrievalResult(
                query=query,
                seed_entities=seed_entities,
                candidate_clip_count=0,
            )

        # Step 4 — Hybrid ranking (BM25 + Dense Semantic)
        # Fetch more candidates initially to allow for pruning during verification
        ranked_clips = self._rank_bm25(query, query_vec, filtered_candidates, top_n_clips * 3)

        # Step 5 — Structured Verification Pipeline
        verified_clips = []
        query_keywords = set(_tokenize(query))
        
        for clip in ranked_clips:
            # 1. Lexical filtering (cheap check)
            clip_text = " ".join(filter(None, [clip.transcript, clip.summary, clip.ocr]))
            clip_tokens = set(_tokenize(clip_text))
            lexical_match = any(kw in clip_tokens for kw in query_keywords)
            
            # 2. Semantic similarity filtering
            semantic_match = clip.semantic_score >= semantic_verification_threshold
            
            # 3. Final Pruning: Clip must have either a strong semantic match or a lexical match
            if lexical_match or semantic_match:
                verified_clips.append(clip)
            
            if len(verified_clips) >= top_n_clips:
                break

        return RetrievalResult(
            query=query,
            seed_entities=seed_entities,
            candidate_clip_count=len(filtered_candidates),
            clips=verified_clips,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_seed_entities(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """Vectorised cosine similarity → top-k entity nodes."""
        if self._entity_embeddings.shape[0] == 0:
            return []

        # Dot product on L2-normalised vectors == cosine similarity
        scores = self._entity_embeddings @ query_vec          # shape (N,)

        # Partial argsort is O(N + k·log k) – faster than full sort for large N
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]  # descending

        seeds = [
            (self._entity_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0   # discard non-positive similarities
        ]
        logger.info("Seed entities: %s", seeds)
        return seeds

    def _traverse_to_clips(
        self, seeds: List[Tuple[str, float]], max_hops: int = 2, 
        semantic_decay: float = 0.6, temporal_decay: float = 0.9
    ) -> Dict[str, float]:
        """Follow bipartite mapping from seed entities to unique clip nodes. 
        Then expands via clip-similarity dict (SHARES_ENTITY) and NEXT edges with score decay."""
        candidate_dict: Dict[str, float] = {}

        # 0. High-precision initial candidate set (Entity -> Clip)
        initial_clips = {}
        for entity_id, conf in seeds:
            # Prefer SQLite mapping dict; fall back to in-graph edges
            if entity_id in self.bipartite_dict:
                for clip_info in self.bipartite_dict[entity_id]:
                    cid = clip_info['clip_id']
                    c_conf = clip_info.get('confidence', 1.0) * conf
                    if cid not in initial_clips or c_conf > initial_clips[cid]:
                        initial_clips[cid] = c_conf
            else:
                for neighbour in self.graph.successors(entity_id):
                    if neighbour in self._clip_ids:
                        edge_data = self.graph.get_edge_data(entity_id, neighbour, default={})
                        c_conf = edge_data.get('confidence', 1.0) * conf
                        if neighbour not in initial_clips or c_conf > initial_clips[neighbour]:
                            initial_clips[neighbour] = c_conf
                for neighbour in self.graph.predecessors(entity_id):
                    if neighbour in self._clip_ids:
                        edge_data = self.graph.get_edge_data(neighbour, entity_id, default={})
                        c_conf = edge_data.get('confidence', 1.0) * conf
                        if neighbour not in initial_clips or c_conf > initial_clips[neighbour]:
                            initial_clips[neighbour] = c_conf

        candidate_dict.update(initial_clips)

        # 1. Edge-Type-Aware Traversal (Expansion)
        current_frontier = list(initial_clips.items())
        
        for hop in range(max_hops):
            next_frontier = {}
            for cid, score in current_frontier:

                # ── SHARES_ENTITY via SQLite similarity dict ───────────────
                for sim_entry in self._clip_similarity_dict.get(cid, []):
                    neighbour = sim_entry['clip_id']
                    new_score = score * semantic_decay
                    if neighbour not in candidate_dict or new_score > candidate_dict[neighbour]:
                        candidate_dict[neighbour] = new_score
                        next_frontier[neighbour] = new_score

                # ── NEXT edges via in-graph traversal ─────────────────────
                for neighbour in self.graph.successors(cid):
                    if neighbour in self._clip_ids:
                        edge_data = self.graph.get_edge_data(cid, neighbour, default={})
                        edge_type = edge_data.get('type')
                        if edge_type != 'NEXT':
                            continue
                        new_score = score * temporal_decay
                        if neighbour not in candidate_dict or new_score > candidate_dict[neighbour]:
                            candidate_dict[neighbour] = new_score
                            next_frontier[neighbour] = new_score
                
                for neighbour in self.graph.predecessors(cid):
                    if neighbour in self._clip_ids:
                        edge_data = self.graph.get_edge_data(neighbour, cid, default={})
                        edge_type = edge_data.get('type')
                        if edge_type != 'NEXT':
                            continue
                        new_score = score * temporal_decay * 0.8  # penalty for going backwards
                        if neighbour not in candidate_dict or new_score > candidate_dict[neighbour]:
                            candidate_dict[neighbour] = new_score
                            next_frontier[neighbour] = new_score
            
            current_frontier = list(next_frontier.items())
            if not current_frontier:
                break

        logger.info(
            "Graph traversal yielded %d candidate clips from %d seed entities (max_hops=%d).",
            len(candidate_dict), len(seeds), max_hops
        )
        return candidate_dict

    def _rank_bm25(
        self,
        query: str,
        query_vec: np.ndarray,
        candidate_dict: Dict[str, float],
        top_n: int,
    ) -> List[RetrievedClip]:
        """Hybrid scoring combining BM25Okapi, Dense Semantic Similarity, and Entity Confidence."""
        candidate_ids = list(candidate_dict.keys())
        
        # Build corpus (tokenised docs)
        corpus_tokens: List[List[str]] = []
        candidate_texts: List[str] = []
        for cid in candidate_ids:
            attr = self.graph.nodes[cid]
            doc_text = " ".join(filter(None, [
                attr.get("transcript", ""),
                attr.get("ocr", ""),
                attr.get("keywords", ""),
                attr.get("summary", ""),
            ]))
            corpus_tokens.append(_tokenize(doc_text))
            candidate_texts.append(doc_text)

        bm25 = BM25Plus(corpus_tokens)
        query_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)   # ndarray (len(candidates),)

        # Compute Dense Semantic Similarity for candidates
        semantic_embeddings = self._model.encode(
            candidate_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        semantic_scores = semantic_embeddings @ query_vec

        # Combine BM25, Semantic Score, and Entity Confidence
        final_scores = np.zeros_like(bm25_scores)
        for i, cid in enumerate(candidate_ids):
            conf = candidate_dict[cid]
            # Final score: Weight semantic heavily to solve synonymous matches
            final_scores[i] = (bm25_scores[i] * 0.5) + (semantic_scores[i] * 8.0) + (conf * 2.0)

        # Top-n by score
        n = min(top_n, len(final_scores))
        top_indices = np.argpartition(final_scores, -n)[-n:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

        results: List[RetrievedClip] = []
        for idx in top_indices:
            cid = candidate_ids[idx]
            attr = self.graph.nodes[cid]
            results.append(
                RetrievedClip(
                    node_id=cid,
                    video_id=attr.get("video_id", ""),
                    start=attr.get("start", 0.0),
                    end=attr.get("end", 0.0),
                    bm25_score=float(bm25_scores[idx]),
                    entity_confidence=float(candidate_dict[cid]),
                    semantic_score=float(semantic_scores[idx]),
                    final_score=float(final_scores[idx]),
                    transcript=attr.get("transcript", ""),
                    ocr=attr.get("ocr", ""),
                    summary=attr.get("summary", ""),
                )
            )

        return results


# ---------------------------------------------------------------------------
# Convenience function (top-level entry point)
# ---------------------------------------------------------------------------

def retrieve_clips(
    graph: nx.DiGraph,
    query: str,
    top_k_entities: int = 3,
    top_n_clips: int = 5,
    *,
    bipartite_dict: Dict[str, List[Dict[str, Any]]] = None,
    mapping_db_path: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RetrievalResult:
    """
    One-shot convenience wrapper.  Builds the retriever, embeds entities,
    and returns results.  For repeated queries, prefer instantiating
    ``VideoGraphRetriever`` once and calling ``.retrieve()`` in a loop.

    Pass ``mapping_db_path`` to load the bipartite mapping from SQLite,
    or ``bipartite_dict`` for an in-memory fallback.
    """
    retriever = VideoGraphRetriever(
        graph, bipartite_dict,
        model_name=model_name,
        mapping_db_path=mapping_db_path,
    )
    return retriever.retrieve(query, top_k_entities=top_k_entities, top_n_clips=top_n_clips)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_loader import VideoDataLoader
    from temporal_clip_graph import build_temporal_clip_graph
    from semantic_graph import build_semantic_graph, build_bipartite_mapping

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    outputs_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    query_text = sys.argv[2] if len(sys.argv) > 2 else "What brands appear in the video?"

    # 1. Build graph
    loader = VideoDataLoader(outputs_dir)
    data = loader.load_data()
    G = build_temporal_clip_graph(data)
    G = build_semantic_graph(G, data)
    bipartite_dict = build_bipartite_mapping(G, data)

    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # 2. Retrieve
    retriever = VideoGraphRetriever(G, bipartite_dict)
    result = retriever.retrieve(query_text)

    # 3. Display results
    print(f"\n{'='*70}")
    print(f"  Query: {result.query}")
    print(f"  Seed entities: {result.seed_entities}")
    print(f"  Candidate clips evaluated: {result.candidate_clip_count}")
    print(f"{'='*70}\n")

    for i, clip in enumerate(result.clips, 1):
        print(f"  #{i}  [{clip.video_id}]  {clip.start:.2f}s – {clip.end:.2f}s")
        print(f"       Final Score : {clip.final_score:.4f} (BM25: {clip.bm25_score:.4f}, Entity Conf: {clip.entity_confidence:.4f})")
        print(f"       Transcript  : {clip.transcript[:120]}...")
        print(f"       OCR         : {clip.ocr[:80]}...")
        print()
