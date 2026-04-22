import re
from typing import Dict, List, Tuple
from collections import Counter
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

class SemanticNode(BaseModel):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    embedding: List[float]
    support_count: int

# ---------------------------------------------------------------------------
# UTILITIES (Reimplemented for Self-Containment and Determinism)
# ---------------------------------------------------------------------------

def normalize_entity(text: str) -> str:
    """Lowercase, strip whitespace, and collapse spaces."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def embed_text(text: str) -> List[float]:
    """
    Deterministic embedding function (simulating Stage 3).
    Uses normalized character frequency to generate a deterministic vector
    so that similar strings naturally have higher cosine similarity.
    """
    vec = [0.0] * 26
    for char in normalize_entity(text):
        if 'a' <= char <= 'z':
            vec[ord(char) - ord('a')] += 1.0
    
    # Normalize vector to unit length
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return [0.0] * 26
    return [v / norm for v in vec]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def update_centroid(embeddings: List[List[float]]) -> List[float]:
    """Calculate the mean embedding for a cluster."""
    if not embeddings:
        return []
    num_dims = len(embeddings[0])
    num_vecs = len(embeddings)
    centroid = [sum(emb[i] for emb in embeddings) / num_vecs for i in range(num_dims)]
    return centroid

# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

class EntityCanonicalizer:
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def process(self, clip_entity_map: Dict[str, List[str]]) -> Tuple[List[SemanticNode], Dict[str, List[str]]]:
        if not clip_entity_map:
            return [], {}

        # 1. Flatten Entities & Count Frequencies (of normalized strings)
        # We track occurrences of both raw and normalized to pick the best canonical name.
        normalized_freq = Counter()
        raw_to_norm = {}
        
        for clip_id, raw_entities in clip_entity_map.items():
            for raw_ent in raw_entities:
                norm_ent = normalize_entity(raw_ent)
                if not norm_ent:
                    continue
                normalized_freq[norm_ent] += 1
                raw_to_norm[raw_ent] = norm_ent

        unique_normalized = sorted(list(normalized_freq.keys()))

        # 2. Embedding caching
        embeddings_cache = {ent: embed_text(ent) for ent in unique_normalized}

        # 3. Embedding-Based Clustering
        # List of clusters. Each cluster is a dict:
        # { 'entities': list[str], 'centroid': list[float], 'embeddings': list[list[float]] }
        clusters = []

        for entity in unique_normalized:
            emb = embeddings_cache[entity]
            
            best_match_idx = -1
            best_sim = -1.0
            
            for i, cluster in enumerate(clusters):
                sim = cosine_similarity(emb, cluster['centroid'])
                if sim > best_sim:
                    best_sim = sim
                    best_match_idx = i
            
            if best_sim > self.similarity_threshold:
                # Add to existing cluster
                clusters[best_match_idx]['entities'].append(entity)
                clusters[best_match_idx]['embeddings'].append(emb)
                # Update centroid
                clusters[best_match_idx]['centroid'] = update_centroid(clusters[best_match_idx]['embeddings'])
            else:
                # Create new cluster
                clusters.append({
                    'entities': [entity],
                    'centroid': emb,
                    'embeddings': [emb]
                })

        # 4. & 5. & 6. & 7. Build SemanticNodes
        semantic_nodes = []
        norm_to_concept = {}

        for index, cluster in enumerate(clusters):
            cluster_entities = sorted(cluster['entities'])
            
            # Select canonical name: most frequent, tie-break longest string, then alphabetical
            canonical_name = max(
                cluster_entities,
                key=lambda e: (normalized_freq[e], len(e), e)
            )
            
            concept_id = f"concept_{index}"
            support_count = sum(normalized_freq[e] for e in cluster_entities)
            
            node = SemanticNode(
                concept_id=concept_id,
                canonical_name=canonical_name,
                aliases=cluster_entities,
                embedding=cluster['centroid'],
                support_count=support_count
            )
            semantic_nodes.append(node)
            
            # Map normalized entities to the new concept ID
            for e in cluster_entities:
                norm_to_concept[e] = concept_id

        # 8. Clip Mapping
        clip_to_concepts = {}
        for clip_id, raw_entities in sorted(clip_entity_map.items()):
            concept_ids = set()
            for raw_ent in raw_entities:
                norm_ent = normalize_entity(raw_ent)
                if norm_ent in norm_to_concept:
                    concept_ids.add(norm_to_concept[norm_ent])
            
            # Deduplicate and sort for determinism
            clip_to_concepts[clip_id] = sorted(list(concept_ids))

        return semantic_nodes, clip_to_concepts


# ---------------------------------------------------------------------------
# TESTING REQUIREMENT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock entity input
    mock_input = {
        "clip_001": ["coffee cup", "cup", "kitchen"],
        "clip_002": ["mug", "coffee", "kitchen", "KITCHEN  "],
        "clip_003": ["tea cup", "coffee cup", "living room"],
        "clip_004": ["empty clip test"]
    }

    print("--- Running Entity Canonicalizer ---")
    canonicalizer = EntityCanonicalizer(similarity_threshold=0.8)
    nodes, clip_map = canonicalizer.process(mock_input)

    print("\n[Semantic Nodes]")
    for node in nodes:
        print(f"ID: {node.concept_id}")
        print(f"  Canonical Name: '{node.canonical_name}'")
        print(f"  Aliases: {node.aliases}")
        print(f"  Support Count: {node.support_count}")
        print(f"  Embedding Dim: {len(node.embedding)}")

    print("\n[Clip to Concept Mapping]")
    for clip, concepts in clip_map.items():
        print(f"{clip} -> {concepts}")

# ---------------------------------------------------------------------------
# FUNCTION-WRAPPERS FOR PIPELINE
# ---------------------------------------------------------------------------

def canonicalize_entities(
    clip_entity_map: Dict[str, List[str]],
    similarity_threshold: float = 0.8
) -> Tuple[List[SemanticNode], Dict[str, List[str]]]:
    """Function-style API expected by pipeline.py."""
    return EntityCanonicalizer(similarity_threshold=similarity_threshold).process(clip_entity_map)

def canonicalize(
    clip_entity_map: Dict[str, List[str]],
    similarity_threshold: float = 0.8
) -> Tuple[List[SemanticNode], Dict[str, List[str]]]:
    """Alias for compatibility."""
    return canonicalize_entities(clip_entity_map, similarity_threshold=similarity_threshold)

def build_canonical_entities(
    clip_entity_map: Dict[str, List[str]],
    similarity_threshold: float = 0.8
) -> Tuple[List[SemanticNode], Dict[str, List[str]]]:
    """Alias for compatibility."""
    return canonicalize_entities(clip_entity_map, similarity_threshold=similarity_threshold)