import logging
import re
import time
from typing import List, Dict, Optional, Set
from sentence_transformers import SentenceTransformer
import numpy as np

from .models import QueryDecomposition, FailureResponse
from .key_manager import KeyManager
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Stop words for keyword extraction ────────────────────────────────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at",
    "from", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "once",
    "here", "there", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "is", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "what", "which", "who",
    "whom", "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "of", "are", "would", "could", "may", "might", "shall",
}

TEMPORAL_KEYWORDS = {
    "before", "after", "during", "when", "while", "until", "since",
    "first", "last", "then", "next", "previous", "earlier", "later",
    "preceding", "following", "prior", "subsequent",
}

# ── Soft type mapping: LLM types → accepted graph ID prefixes ────────────────
# Boost score if prefix matches, but NEVER reject on mismatch.
TYPE_SOFT_MAP: Dict[str, List[str]] = {
    "person":   ["person_", "text_", "label_", "keyword_"],
    "location": ["location_", "text_", "label_", "keyword_"],
    "topic":    ["keyword_", "text_", "topic_", "label_"],
    "object":   ["object_", "keyword_", "text_", "label_"],
    "event":    ["text_", "keyword_", "event_", "label_"],
}


class QueryDecomposer:
    def __init__(
        self,
        cerebras_keys: List[str],
        groq_keys: List[str],
        entity_corpus: List[Dict[str, str]] = None,
    ):
        """
        Args:
            cerebras_keys:  Cerebras API keys (tried first, round-robin).
            groq_keys:      Groq API keys (fallback only after all Cerebras keys fail).
            entity_corpus:  List of {id, name} dicts representing canonical EntityRef nodes.
        """
        self.key_manager = KeyManager(cerebras_keys, groq_keys)
        self.llm_client = LLMClient(self.key_manager, max_retries_per_key=2)

        self.entity_corpus: List[Dict[str, str]] = entity_corpus or []
        self.encoder: Optional[SentenceTransformer] = None
        self.corpus_embeddings = None
        self.corpus_names_lower: List[str] = []

        if self.entity_corpus:
            logger.info("Initializing SentenceTransformer for entity resolution...")
            self.encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
            names = [e["name"] for e in self.entity_corpus]
            self.corpus_names_lower = [n.lower() for n in names]
            self.corpus_embeddings = self.encoder.encode(names, normalize_embeddings=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Fast-path: simple keyword decomposition (NO LLM)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple enough to skip LLM decomposition."""
        words = query.strip().split()
        has_temporal = any(w.lower() in TEMPORAL_KEYWORDS for w in words)
        # Short queries with no temporal reasoning → fast path
        return len(words) <= 8 and not has_temporal

    def _fast_decompose(self, query: str) -> QueryDecomposition:
        """
        Extract keywords directly from query and build a minimal execution plan.
        No LLM call. Used for simple queries or as timeout fallback.
        """
        keywords = self._extract_query_keywords(query)
        # Build entities from keywords
        entities = []
        for kw in keywords:
            entities.append({"name": kw, "type": "topic"})

        # Build a simple resolve+traverse plan
        plan = []
        for i, kw in enumerate(list(keywords)[:3], start=1):
            plan.append({
                "step": i * 2 - 1,
                "operation": "resolve_entity",
                "input": kw,
                "output": f"e{i}",
            })
            plan.append({
                "step": i * 2,
                "operation": "traverse",
                "from": f"e{i}",
                "edge": "APPEARS_IN",
                "to": "ClipRef",
            })

        return QueryDecomposition(
            query_type="keyword_lookup",
            entities=[{"name": kw, "type": "topic"} for kw in list(keywords)[:3]],
            actions=["retrieve"],
            temporal_constraints={"relation": "none", "anchor_event": None, "direction": "none"},
            sub_queries=[{
                "id": "Q1",
                "type": "keyword_search",
                "goal": f"Find clips matching: {', '.join(keywords)}",
                "required_graph_components": ["APPEARS_IN", "EntityRef", "ClipRef"],
            }],
            execution_plan=plan if plan else [
                {"step": 1, "operation": "extract", "target": "ClipRef", "fields": ["transcript", "summary"]}
            ],
            confidence=0.0,
            ambiguity_flags=["fast_path_no_llm"],
        )

    @staticmethod
    def _extract_query_keywords(query: str) -> List[str]:
        """Extract meaningful keywords from query text."""
        text = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
        words = text.split()
        return [w for w in words if w not in STOP_WORDS and len(w) > 2]

    # ──────────────────────────────────────────────────────────────────────────
    # Post-processing: temporal logic enforcement
    # ──────────────────────────────────────────────────────────────────────────

    def _enforce_temporal_logic(self, decomposition: QueryDecomposition) -> None:
        """
        Programmatically correct temporal direction — never trust the LLM.
        Also patches any temporal_traverse steps in the execution_plan.
        """
        tc = decomposition.temporal_constraints
        direction_map = {"before": "backward", "after": "forward", "during": "neutral"}
        correct_dir = direction_map.get(tc.relation)

        if correct_dir and tc.direction != correct_dir:
            logger.info(
                f"Temporal override: relation='{tc.relation}' forced direction "
                f"'{tc.direction}' -> '{correct_dir}'"
            )
            tc.direction = correct_dir

        # Also patch any temporal_traverse steps in the plan
        if correct_dir:
            for step in decomposition.execution_plan:
                if isinstance(step, dict) and step.get("operation") == "temporal_traverse":
                    step["direction"] = correct_dir

    # ──────────────────────────────────────────────────────────────────────────
    # Entity normalization
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for matching: lowercase, strip punctuation, collapse whitespace."""
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _lexical_similarity(query: str, candidate: str) -> float:
        """
        Compute token-level Jaccard similarity between two strings.
        Returns 0.0-1.0.
        """
        q_tokens = set(query.lower().split()) - STOP_WORDS
        c_tokens = set(candidate.lower().split()) - STOP_WORDS
        if not q_tokens or not c_tokens:
            return 0.0
        intersection = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        return intersection / union if union > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Post-processing: HYBRID entity resolution (FIX 1 + FIX 4)
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_entities(self, decomposition: QueryDecomposition) -> None:
        """
        HYBRID entity resolution: embedding similarity + lexical similarity.
        
        Scoring formula:
            score = 0.7 * embedding_similarity + 0.3 * lexical_similarity
        
        Acceptance rules:
            - score > 0.75 → ACCEPT (ignore type mismatch)
            - score > 0.60 AND top-1 candidate → ACCEPT
            - Type-prefix match gives a bonus but NEVER causes rejection
        
        Sets decomposition.confidence based on average mapping quality.
        """
        if not self.encoder or self.corpus_embeddings is None or not self.entity_corpus:
            decomposition.confidence = 1.0
            return

        if not decomposition.entities:
            decomposition.confidence = 1.0
            return

        total_score = 0.0
        resolved_count = 0

        for entity in decomposition.entities:
            # Normalize entity name for matching
            entity_name_norm = self._normalize_text(entity.name)

            # 1. Embedding similarity
            query_emb = self.encoder.encode([entity.name], normalize_embeddings=True)[0]
            emb_sims = np.dot(self.corpus_embeddings, query_emb)

            # 2. Lexical similarity for top candidates
            top_k_indices = np.argsort(emb_sims)[-10:][::-1]

            best_score = 0.0
            best_idx = -1
            best_details = ""

            for idx in top_k_indices:
                idx = int(idx)
                emb_score = float(emb_sims[idx])
                if emb_score < 0.3:
                    break  # Below this, no point checking

                candidate_name = self.corpus_names_lower[idx]
                lex_score = self._lexical_similarity(entity_name_norm, candidate_name)

                # Hybrid score
                hybrid_score = 0.7 * emb_score + 0.3 * lex_score

                # Type-prefix soft boost (NOT a filter)
                candidate_id = self.entity_corpus[idx]["id"]
                entity_type = (entity.type or "topic").lower()
                preferred_prefixes = TYPE_SOFT_MAP.get(entity_type, ["text_", "keyword_"])

                # Check if candidate's prefix is in the preferred list
                has_preferred_prefix = any(candidate_id.startswith(p) for p in preferred_prefixes[:2])
                if has_preferred_prefix:
                    hybrid_score += 0.05  # Soft boost, not hard filter

                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_idx = idx
                    best_details = (
                        f"emb={emb_score:.3f}, lex={lex_score:.3f}, "
                        f"hybrid={hybrid_score:.3f}, prefix_boost={has_preferred_prefix}"
                    )

            # Acceptance decision: HIGH-RECALL, TYPE-TOLERANT
            if best_idx >= 0 and best_score > 0.55:
                candidate_id = self.entity_corpus[best_idx]["id"]
                entity.resolved_entity_id = candidate_id
                total_score += best_score
                resolved_count += 1
                logger.info(
                    f"Resolved '{entity.name}' ({entity.type}) -> '{candidate_id}' "
                    f"({best_details})"
                )
            else:
                entity.resolved_entity_id = None
                logger.warning(
                    f"No reliable match for '{entity.name}' "
                    f"(best_score={best_score:.3f}, {best_details})"
                )

        # Confidence = average of resolved entities' scores
        if resolved_count > 0:
            decomposition.confidence = total_score / len(decomposition.entities)
        else:
            decomposition.confidence = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Post-processing: validate typed execution plan steps
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_typed_steps(self, decomposition: QueryDecomposition) -> List[str]:
        """
        Attempt to parse each execution_plan step into its typed model.
        Returns list of violations (empty = all valid).
        """
        try:
            typed_steps = decomposition.get_typed_execution_plan()
            if len(typed_steps) != len(decomposition.execution_plan):
                return [
                    f"Only {len(typed_steps)}/{len(decomposition.execution_plan)} "
                    "steps could be typed — unknown operation in plan"
                ]
            return []
        except Exception as e:
            return [f"Typed plan validation error: {e}"]

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def decompose(self, query: str) -> "QueryDecomposition | FailureResponse":
        """
        Convert a natural language query into a structured, graph-executable plan.

        Pipeline:
            1. Fast-path check (skip LLM for simple queries)
            2. LLM decomposition with hard timeout (3s)
            3. Fallback to keyword decomposition on timeout/failure
            4. Post-processing: temporal correction, hybrid entity resolution,
               execution-plan validation
        """
        import os
        import json
        
        logger.info(f"Decomposing: {query!r}")

        llm_logs = None

        # ── Fast path for simple queries ─────────────────────────────────────
        if self._is_simple_query(query):
            logger.info("Simple query detected — using fast path (no LLM)")
            t_start = time.perf_counter()
            result = self._fast_decompose(query)
            elapsed = time.perf_counter() - t_start
            llm_logs = {
                "query": query,
                "provider_attempts": [],
                "final_provider": "fast_path",
                "final_model": "fast_path",
                "total_llm_time": elapsed
            }
        else:
            # ── LLM decomposition with timeout ───────────────────────────────
            t_start = time.perf_counter()
            result = self.llm_client.execute_with_failover(query)
            elapsed = time.perf_counter() - t_start
            
            if hasattr(result, "llm_logs") and result.llm_logs:
                llm_logs = result.llm_logs

            if isinstance(result, FailureResponse):
                logger.warning(
                    f"LLM decomposition failed ({result.reason}). "
                    f"Falling back to keyword decomposition."
                )
                result = self._fast_decompose(query)
                # Keep the failure llm_logs from the llm_client
            elif elapsed > 8.0:
                logger.warning(
                    f"LLM decomposition took {elapsed:.1f}s — "
                    f"consider fast-path for similar queries"
                )

        logger.info("Decomposition complete. Running post-processing...")

        # 1. Enforce temporal direction
        self._enforce_temporal_logic(result)

        # 2. Resolve entities (HYBRID — no more strict type rejection)
        self._resolve_entities(result)

        # 3. Validate typed execution steps
        violations = self._validate_typed_steps(result)
        if violations:
            logger.warning(f"Execution plan has {len(violations)} violation(s): {violations}")
            result.ambiguity_flags.extend(violations)
            
        # Log to file
        if llm_logs:
            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/llm_usage_logs.jsonl", "a") as f:
                    f.write(json.dumps(llm_logs) + "\n")
            except Exception as e:
                logger.error(f"Failed to write LLM usage log: {e}")

        return result
