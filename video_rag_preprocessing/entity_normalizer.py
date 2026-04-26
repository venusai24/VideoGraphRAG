import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EntityNormalizer:
    """
    Robust Entity Normalization using bucketed embedding similarity.
    Maintains a persistent canonical entity mapping.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", merge_threshold=0.85):
        logger.info(f"Initializing EntityNormalizer with {model_name} (threshold: {merge_threshold})")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.merge_threshold = merge_threshold
        self.canonical_mapping: Dict[str, str] = {}
        self.prototype_entities: Dict[str, Dict[str, Any]] = {}

    def get_canonical_id(self, raw_id: str) -> str:
        """Return the canonical ID for a raw entity ID, or the raw ID if unmerged."""
        return self.canonical_mapping.get(raw_id, raw_id)

    def normalize_entities(self, raw_entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Takes a list of raw entity dictionaries (must have 'id', 'type', 'name', 'description').
        Computes embeddings bucketed by 'type', merges highly similar entities,
        and returns the updated canonical mapping.
        """
        # 1. Bucket entities by type to avoid O(N^2) across unrelated types
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for ent in raw_entities:
            if ent['id'] in self.canonical_mapping:
                continue # Already normalized
            etype = ent.get('type', 'unknown')
            if etype not in buckets:
                buckets[etype] = []
            buckets[etype].append(ent)

        new_mappings_count = 0
        
        # 2. Process each bucket
        for etype, entities in buckets.items():
            if len(entities) < 2:
                for ent in entities:
                    self.canonical_mapping[ent['id']] = ent['id']
                    self.prototype_entities[ent['id']] = ent
                continue

            # Build text for embedding
            texts = []
            for ent in entities:
                name = str(ent.get('name', ''))
                desc = str(ent.get('description', ''))
                texts.append(f"{name} {desc}".strip())

            # Compute embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

            # 3. Simple Greedy Clustering for Normalization
            n = len(entities)
            merged = [False] * n
            
            for i in range(n):
                if merged[i]: continue
                
                canonical_ent = entities[i]
                canonical_id = canonical_ent['id']
                
                self.canonical_mapping[canonical_id] = canonical_id
                self.prototype_entities[canonical_id] = canonical_ent
                
                # Compare against all others in the bucket
                sims = np.dot(embeddings[i+1:], embeddings[i])
                
                for j_offset, sim in enumerate(sims):
                    j = i + 1 + j_offset
                    if not merged[j] and sim >= self.merge_threshold:
                        # Merge j into i
                        duplicate_id = entities[j]['id']
                        self.canonical_mapping[duplicate_id] = canonical_id
                        merged[j] = True
                        new_mappings_count += 1
                        
                        # Merge descriptions if canonical lacks one
                        if not self.prototype_entities[canonical_id].get('description') and entities[j].get('description'):
                            self.prototype_entities[canonical_id]['description'] = entities[j]['description']
                            
                merged[i] = True

        logger.info(f"Entity Normalization: Merged {new_mappings_count} entities across {len(buckets)} types.")
        return self.canonical_mapping
