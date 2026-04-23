import re
import string
import spacy
from spacy.cli import download as spacy_download
from typing import Dict, List, Set, Any
from pydantic import BaseModel
import json

# Schema Definition
class ClipNode(BaseModel):
    clip_id: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    transcript_text: str
    ocr_text: str
    keywords: List[str]
    evidence_ids: List[str]

# Global singleton for the spaCy model
_NLP_MODEL = None

def get_nlp() -> Any:
    """
    Loads and returns the spaCy NLP model. Uses a singleton pattern
    to ensure the model is loaded only once per session.
    """
    global _NLP_MODEL
    if _NLP_MODEL is None:
        try:
            _NLP_MODEL = spacy.load("en_core_web_sm")
        except OSError:
            spacy_download("en_core_web_sm")
            _NLP_MODEL = spacy.load("en_core_web_sm")
    return _NLP_MODEL

def normalize_entity(text: str) -> str:
    """
    Normalizes the extracted entity strings.
    Rules: lowercase, strip whitespace, remove boundary punctuation, 
    collapse spaces, remove < 2 char tokens.
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Strip whitespace and punctuation from boundaries
    text = text.strip(string.punctuation + " \t\n\r")
    
    # Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short tokens
    if len(text) < 2:
        return ""
        
    return text

def is_valid_entity(text: str, stop_words: Set[str]) -> bool:
    """
    Applies light filtering to the normalized entities.
    Removes pure stopwords and numeric-only strings.
    Keeps short but meaningful words.
    """
    if not text:
        return False
        
    # Remove numeric-only strings
    if text.isnumeric():
        return False
        
    # Remove pure stopwords (matches exact normalized string)
    if text in stop_words:
        return False
        
    return True

def extract_from_clip(clip: ClipNode, nlp: Any) -> List[str]:
    """
    Extracts entities from a single ClipNode using multi-strategy extraction.
    """
    candidates: List[str] = []
    
    # 1. Keyword Inclusion (High-value signals)
    if clip.keywords:
        candidates.extend(clip.keywords)
        
    # 2. Text Combination
    combined_text = f"{clip.transcript_text} {clip.ocr_text}".strip()
    
    if combined_text:
        doc = nlp(combined_text)
        
        # 3. Named Entity Recognition (NER)
        target_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}
        for ent in doc.ents:
            if ent.label_ in target_labels:
                candidates.append(ent.text)
                
        # 4. Noun Phrase Extraction
        for chunk in doc.noun_chunks:
            candidates.append(chunk.text)
            
    # 5. Normalization, Filtering, and Deduplication (preserving order)
    seen: set = set()
    results: List[str] = []
    stop_words = nlp.Defaults.stop_words
    
    for candidate in candidates:
        normalized = normalize_entity(candidate)
        if is_valid_entity(normalized, stop_words):
            if normalized not in seen:
                seen.add(normalized)
                results.append(normalized)
                
    return results

def extract_entities(clips: List[ClipNode]) -> Dict[str, List[str]]:
    """
    Main entry point. Takes a list of ClipNodes and returns a mapping
    of clip_id to a list of extracted and normalized entity strings.
    """
    nlp = get_nlp()
    output: Dict[str, List[str]] = {}
    
    for clip in clips:
        entities = extract_from_clip(clip, nlp)
        output[clip.clip_id] = entities
        
    return output

if __name__ == "__main__":
    # Mock data generation
    mock_clips = [
        ClipNode(
            clip_id="clip_001",
            video_id="vid_x1",
            start_time_ms=0,
            end_time_ms=5000,
            transcript_text="John Doe walked into the kitchen and placed the coffee cup on the wooden table.",
            ocr_text="12345 CAFE COFFEE",
            keywords=["morning routine", "coffee time"],
            evidence_ids=["ev_1"]
        ),
        ClipNode(
            clip_id="clip_002",
            video_id="vid_x1",
            start_time_ms=5000,
            end_time_ms=10000,
            transcript_text="He opened the door.",
            ocr_text="  ",
            keywords=["door", "exit"],
            evidence_ids=["ev_2"]
        ),
        ClipNode(
            clip_id="clip_003",
            video_id="vid_x1",
            start_time_ms=10000,
            end_time_ms=15000,
            transcript_text="",
            ocr_text="   ",
            keywords=[],
            evidence_ids=[]
        )
    ]
    
    # Run extraction
    extracted_data = extract_entities(mock_clips)
    
    # Print results deterministically
    print(json.dumps(extracted_data, indent=4))