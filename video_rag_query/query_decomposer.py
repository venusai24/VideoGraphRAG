import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from .models import QueryDecomposition, FailureResponse
from .key_manager import KeyManager
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


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
        self.llm_client = LLMClient(self.key_manager, max_retries_per_key=3)

        self.entity_corpus: List[Dict[str, str]] = entity_corpus or []
        self.encoder: Optional[SentenceTransformer] = None
        self.corpus_embeddings = None

        if self.entity_corpus:
            logger.info("Initializing SentenceTransformer for entity resolution...")
            self.encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
            names = [e["name"] for e in self.entity_corpus]
            self.corpus_embeddings = self.encoder.encode(names, normalize_embeddings=True)

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
    # Post-processing: entity resolution
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_entities(self, decomposition: QueryDecomposition) -> None:
        """
        Map extracted entity names to canonical EntityRef IDs via embedding similarity.
        Enforces strict type-prefix consistency; mismatches are nulled out, not silently accepted.
        Sets decomposition.confidence based on average mapping quality.
        """
        if not self.encoder or self.corpus_embeddings is None or not self.entity_corpus:
            decomposition.confidence = 1.0
            return

        if not decomposition.entities:
            decomposition.confidence = 1.0
            return

        total_score = 0.0

        for entity in decomposition.entities:
            query_emb = self.encoder.encode([entity.name], normalize_embeddings=True)[0]
            sims = np.dot(self.corpus_embeddings, query_emb)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            if best_score > 0.6:
                candidate_id = self.entity_corpus[best_idx]["id"]
                expected_prefix = f"{entity.type.lower()}_"

                if candidate_id.startswith(expected_prefix):
                    entity.resolved_entity_id = candidate_id
                    total_score += best_score
                    logger.info(
                        f"Resolved '{entity.name}' ({entity.type}) -> '{candidate_id}' "
                        f"(score={best_score:.2f})"
                    )
                else:
                    entity.resolved_entity_id = None
                    logger.warning(
                        f"Rejected mapping for '{entity.name}': type mismatch "
                        f"(expected prefix '{expected_prefix}', got '{candidate_id}')"
                    )
            else:
                entity.resolved_entity_id = None
                logger.warning(
                    f"No reliable match for '{entity.name}' (best score={best_score:.2f})"
                )

        decomposition.confidence = total_score / len(decomposition.entities)

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

    def decompose(self, query: str) -> QueryDecomposition | FailureResponse:
        """
        Convert a natural language query into a structured, graph-executable plan.
        Post-processing enforces temporal correctness, entity type safety, and
        execution-plan structural validity.
        """
        logger.info(f"Decomposing: {query!r}")
        result = self.llm_client.execute_with_failover(query)

        if not isinstance(result, QueryDecomposition):
            logger.error(f"Decomposition failed: {result.reason}")
            return result

        logger.info("LLM response validated. Running post-processing...")

        # 1. Enforce temporal direction
        self._enforce_temporal_logic(result)

        # 2. Resolve entities
        self._resolve_entities(result)

        # 3. Validate typed execution steps
        violations = self._validate_typed_steps(result)
        if violations:
            logger.warning(f"Execution plan has {len(violations)} violation(s): {violations}")
            result.ambiguity_flags.extend(violations)

        return result
