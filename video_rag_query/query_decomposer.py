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
        """
        if not self.encoder or not self.corpus_embeddings is not None or len(self.entity_corpus) == 0:
            return

        for entity in decomposition.entities:
            query_emb = self.encoder.encode([entity.name], normalize_embeddings=True)[0]
            similarities = np.dot(self.corpus_embeddings, query_emb)
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # Threshold for accepting a match
            if best_score > 0.6:
                entity.resolved_entity_id = self.entity_corpus[best_idx]['id']
                logger.info(f"Mapped entity '{entity.name}' to '{entity.resolved_entity_id}' (score: {best_score:.2f})")
            else:
                logger.warning(f"Could not reliably map entity '{entity.name}' (best score: {best_score:.2f})")

    def decompose(self, query: str) -> QueryDecomposition | FailureResponse:
        """
        Main entry point for query decomposition.
        """
        logger.info(f"Decomposing query: {query}")
        result = self.llm_client.execute_with_failover(query)
        
        if isinstance(result, QueryDecomposition):
            logger.info("Successfully generated decomposition plan. Running post-processing...")
            self._resolve_entities(result)
            return result
        else:
            logger.error(f"Decomposition failed: {result.reason}")
            return result

