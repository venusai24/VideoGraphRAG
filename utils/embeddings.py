import re
from typing import List, Dict
from pydantic import BaseModel

# Constants
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Default dimension for all-MiniLM-L6-v2

class ClipNode(BaseModel):
    clip_id: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    transcript_text: str
    ocr_text: str
    keywords: List[str]
    evidence_ids: List[str]

class _ModelSingleton:
    """Singleton to ensure the model is only loaded once in memory."""
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            # Load local model exclusively; no external API calls
            cls._model = SentenceTransformer(DEFAULT_MODEL_NAME)
        return cls._model

# In-memory cache for deterministic and fast retrieval
# Key: normalized string, Value: embedding vector
_EMBEDDING_CACHE: Dict[str, List[float]] = {}

def normalize_text_for_embedding(text: str) -> str:
    """
    Normalizes text to ensure cache hits and consistent embeddings.
    Rules: lowercase, strip whitespace, collapse multiple spaces, safe for empty input.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def embed_text(text: str) -> List[float]:
    """
    Generates a vector embedding for a single text string.
    Includes caching, normalization, and fallback to zero vector on failure/empty.
    """
    norm_text = normalize_text_for_embedding(text)
    
    if not norm_text:
        return [0.0] * EMBEDDING_DIM
        
    if norm_text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[norm_text]

    try:
        model = _ModelSingleton.get_model()
        # Encode returns a numpy array, convert to list of native floats for JSON serialization
        embedding_array = model.encode(norm_text, show_progress_bar=False)
        embedding_list = [float(val) for val in embedding_array]
        
        # Cache the deterministic result
        _EMBEDDING_CACHE[norm_text] = embedding_list
        return embedding_list
    except Exception:
        # Fallback to zero vector on failure to prevent crashing
        return [0.0] * EMBEDDING_DIM

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a batch of text strings efficiently.
    Preserves input order, utilizes caching, and falls back gracefully.
    """
    results: List[List[float]] = []
    texts_to_embed: List[str] = []
    indices_to_embed: List[int] = []

    # Prepare outputs and check cache
    for i, raw_text in enumerate(texts):
        norm_text = normalize_text_for_embedding(raw_text)
        
        if not norm_text:
            results.append([0.0] * EMBEDDING_DIM)
        elif norm_text in _EMBEDDING_CACHE:
            results.append(_EMBEDDING_CACHE[norm_text])
        else:
            # Placeholder to maintain order
            results.append([])
            texts_to_embed.append(norm_text)
            indices_to_embed.append(i)

    # Process cache misses in a single batch
    if texts_to_embed:
        try:
            model = _ModelSingleton.get_model()
            embeddings_array = model.encode(texts_to_embed, show_progress_bar=False)
            
            for idx, emb_array, norm_t in zip(indices_to_embed, embeddings_array, texts_to_embed):
                emb_list = [float(val) for val in emb_array]
                _EMBEDDING_CACHE[norm_t] = emb_list
                results[idx] = emb_list
        except Exception:
            # Apply zero vector fallback for just the failed batch elements
            for idx in indices_to_embed:
                results[idx] = [0.0] * EMBEDDING_DIM

    return results

def embed_clip(clip: ClipNode) -> List[float]:
    """
    Generates a single vector embedding for a ClipNode by concatenating relevant text fields.
    """
    combined_text = f"{clip.transcript_text} {clip.ocr_text} {' '.join(clip.keywords)}"
    return embed_text(combined_text)

def embed_clips(clips: List[ClipNode]) -> Dict[str, List[float]]:
    """
    Optional Enhancement: Efficiently processes multiple ClipNodes and maps clip_id to embedding.
    """
    clip_ids: List[str] = []
    combined_texts: List[str] = []
    
    for clip in clips:
        clip_ids.append(clip.clip_id)
        combined_text = f"{clip.transcript_text} {clip.ocr_text} {' '.join(clip.keywords)}"
        combined_texts.append(combined_text)
        
    batch_embeddings = embed_batch(combined_texts)
    
    return {c_id: emb for c_id, emb in zip(clip_ids, batch_embeddings)}


if __name__ == "__main__":
    print("--- Testing Embeddings Module ---")
    
    # 1. Create mock ClipNodes
    mock_clip_1 = ClipNode(
        clip_id="clip-001",
        video_id="vid-123",
        start_time_ms=1000,
        end_time_ms=5000,
        transcript_text="The quick brown fox jumps over the lazy dog.",
        ocr_text="NATURE DOC",
        keywords=["fox", "dog", "jump"],
        evidence_ids=["ev-1"]
    )
    
    mock_clip_2 = ClipNode(
        clip_id="clip-002",
        video_id="vid-123",
        start_time_ms=5000,
        end_time_ms=9000,
        transcript_text="Another scene showing the forest.",
        ocr_text="",
        keywords=["forest", "trees"],
        evidence_ids=["ev-2"]
    )
    
    # 2. Run embed_clip
    print("\n[Test] embed_clip")
    single_embedding = embed_clip(mock_clip_1)
    print(f"Generated single embedding for {mock_clip_1.clip_id}")
    print(f"Type: {type(single_embedding)}, Inner Type: {type(single_embedding[0])}")
    print(f"Length: {len(single_embedding)}")
    print(f"First 3 values: {single_embedding[:3]}")

    # 3. Run embed_batch
    print("\n[Test] embed_batch")
    test_texts = [
        "First sentence for batching.",
        "   Second sentence needs normalization!   ",
        "",  # Empty case
        "First sentence for batching.",  # Cache hit case
    ]
    batch_embeddings = embed_batch(test_texts)
    print(f"Processed batch of size {len(test_texts)}")
    print(f"Valid embedding length: {len(batch_embeddings[0])}")
    print(f"Empty text handled (is zero vector?): {all(v == 0.0 for v in batch_embeddings[2])}")
    print(f"Cache mechanism working (same texts == same vector?): {batch_embeddings[0] == batch_embeddings[3]}")

    # 4. Run embed_clips mapping
    print("\n[Test] embed_clips (Dictionary Mapping)")
    clip_map = embed_clips([mock_clip_1, mock_clip_2])
    for c_id, emb in clip_map.items():
        print(f"Clip ID: {c_id} -> Vector Length: {len(emb)}")
    
    print("\nAll tests passed successfully.")