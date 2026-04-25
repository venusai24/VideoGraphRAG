import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from .models import QueryDecomposition, FailureResponse
from .key_manager import KeyManager
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class QueryDecomposer:
    def __init__(self, 
                 cerebras_keys: List[str], 
                 groq_keys: List[str], 
                 entity_corpus: List[Dict[str, str]] = None):
        """
        Args:
            cerebras_keys: List of API keys for Cerebras
            groq_keys: List of API keys for Groq
            entity_corpus: List of dicts with 'id' and 'name' representing canonical EntityRefs in the graph.
        """
        self.key_manager = KeyManager(cerebras_keys, groq_keys)
        self.llm_client = LLMClient(self.key_manager, max_retries_per_key=3)
        
        # Post-processing entity resolution setup
        self.entity_corpus = entity_corpus or []
        self.encoder = None
        self.corpus_embeddings = None
        
        if self.entity_corpus:
            logger.info("Initializing SentenceTransformer for entity resolution...")
            self.encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
            names = [e['name'] for e in self.entity_corpus]
            self.corpus_embeddings = self.encoder.encode(names, normalize_embeddings=True)

    def _resolve_entities(self, decomposition: QueryDecomposition):
        """
        Maps extracted entities to canonical EntityRef IDs via embedding similarity.
        Enforces strict type consistency: if the entity type does not match the corpus type prefix, it is rejected.
        """
        if not self.encoder or not self.corpus_embeddings is not None or len(self.entity_corpus) == 0:
            decomposition.confidence = 1.0 # Base parsing confidence if no corpus
            return

        if not decomposition.entities:
            decomposition.confidence = 1.0 # High confidence if no entities to map
            return

        total_score = 0.0
        mapped_count = 0

        for entity in decomposition.entities:
            query_emb = self.encoder.encode([entity.name], normalize_embeddings=True)[0]
            similarities = np.dot(self.corpus_embeddings, query_emb)
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # Threshold for accepting a match
            if best_score > 0.6:
                best_corpus_id = self.entity_corpus[best_idx]['id']
                # Enforce type consistency (e.g., person -> person_*)
                if best_corpus_id.startswith(f"{entity.type.lower()}_"):
                    entity.resolved_entity_id = best_corpus_id
                    total_score += best_score
                    mapped_count += 1
                    logger.info(f"Mapped entity '{entity.name}' (type: {entity.type}) to '{entity.resolved_entity_id}' (score: {best_score:.2f})")
                else:
                    entity.resolved_entity_id = None
                    total_score += 0.0 # Mismatch counts as 0
                    logger.warning(f"Rejected mapping for '{entity.name}': Type mismatch (expected {entity.type}_*, got {best_corpus_id})")
            else:
                entity.resolved_entity_id = None
                total_score += 0.0
                logger.warning(f"Could not reliably map entity '{entity.name}' (best score: {best_score:.2f})")

        # Compute overall confidence
        decomposition.confidence = total_score / len(decomposition.entities)

    def _enforce_temporal_logic(self, decomposition: QueryDecomposition):
        """
        Programmatically enforce the direction based on the temporal relation.
        DO NOT rely on the LLM for direction correctness.
        """
        if decomposition.temporal_constraints:
            rel = decomposition.temporal_constraints.relation
            if rel == "before":
                decomposition.temporal_constraints.direction = "backward"
            elif rel == "after":
                decomposition.temporal_constraints.direction = "forward"
            elif rel == "during":
                decomposition.temporal_constraints.direction = "neutral"

    def decompose(self, query: str) -> QueryDecomposition | FailureResponse:
        """
        Main entry point for query decomposition.
        """
        logger.info(f"Decomposing query: {query}")
        result = self.llm_client.execute_with_failover(query)
        
        if isinstance(result, QueryDecomposition):
            logger.info("Successfully generated decomposition plan. Running post-processing...")
            self._enforce_temporal_logic(result)
            self._resolve_entities(result)
            return result
        else:
            logger.error(f"Decomposition failed: {result.reason}")
            return result

