import json
import logging
import re
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class EvidenceRecord(BaseModel):
    evidence_id: str
    video_id: str
    source_type: Literal["transcript", "ocr", "keyword"]
    source_item_id: int
    text: str
    normalized_text: str
    confidence: float
    start_time_ms: int
    end_time_ms: int
    speaker_id: Optional[int] = None
    bbox: Optional[Dict[str, int]] = None
    raw_ref: str

def parse_timestamp(ts: str) -> int:
    """
    Parses a timestamp string into milliseconds.
    Handles 'HH:MM:SS.mmm', 'MM:SS.mmm', or 'SS.mmm'.
    Safely returns 0 on malformed values.
    """
    if not ts or not isinstance(ts, str):
        return 0
    try:
        parts = ts.split('.')
        # Handle milliseconds
        ms = int(parts[1][:3].ljust(3, '0')) if len(parts) > 1 else 0
        
        # Handle hours, minutes, seconds
        time_parts = parts[0].split(':')
        seconds = 0
        multiplier = 1
        for part in reversed(time_parts):
            if part.strip():
                seconds += int(part) * multiplier
            multiplier *= 60
            
        return seconds * 1000 + ms
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{ts}': {e}")
        return 0

def normalize_text(text: str) -> str:
    """
    Normalizes text by lowercasing, removing excessive punctuation,
    and stripping extra whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Strip and collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _load_json_safe(file_path: str) -> List[Dict[str, Any]]:
    """Helper to safely load JSON lists with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning(f"File {file_path} does not contain a JSON list.")
            return []
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Malformed JSON in {file_path}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return []

def parse_transcript(file_path: str, video_id: str) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    data = _load_json_safe(file_path)
    
    for item in data:
        try:
            item_id = item.get("id", 0)
            text = item.get("text", "")
            confidence = float(item.get("confidence", 0.0))
            speaker_id = item.get("speakerId")
            instances = item.get("instances", [])
            
            for idx, inst in enumerate(instances):
                start_ts = inst.get("start", "")
                end_ts = inst.get("end", "")
                
                record = EvidenceRecord(
                    evidence_id=f"{video_id}transcript{item_id}_{idx}",
                    video_id=video_id,
                    source_type="transcript",
                    source_item_id=item_id,
                    text=text,
                    normalized_text=normalize_text(text),
                    confidence=confidence,
                    start_time_ms=parse_timestamp(start_ts),
                    end_time_ms=parse_timestamp(end_ts),
                    speaker_id=speaker_id,
                    bbox=None,
                    raw_ref=f"transcript.json#{item_id}"
                )
                records.append(record)
        except ValidationError as ve:
            logger.warning(f"Validation error for transcript item {item.get('id')}: {ve}")
        except Exception as e:
            logger.warning(f"Unexpected error processing transcript item: {e}")
            
    return records

def parse_ocr(file_path: str, video_id: str) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    data = _load_json_safe(file_path)
    
    for item in data:
        try:
            item_id = item.get("id", 0)
            text = item.get("text", "")
            confidence = float(item.get("confidence", 0.0))
            instances = item.get("instances", [])
            
            # Safely extract bounding box if all fields are present
            bbox = None
            if all(k in item for k in ("left", "top", "width", "height", "angle")):
                bbox = {
                    "left": int(item.get("left", 0)),
                    "top": int(item.get("top", 0)),
                    "width": int(item.get("width", 0)),
                    "height": int(item.get("height", 0)),
                    "angle": int(item.get("angle", 0))
                }
            
            for idx, inst in enumerate(instances):
                start_ts = inst.get("start", "")
                end_ts = inst.get("end", "")
                
                record = EvidenceRecord(
                    evidence_id=f"{video_id}ocr{item_id}_{idx}",
                    video_id=video_id,
                    source_type="ocr",
                    source_item_id=item_id,
                    text=text,
                    normalized_text=normalize_text(text),
                    confidence=confidence,
                    start_time_ms=parse_timestamp(start_ts),
                    end_time_ms=parse_timestamp(end_ts),
                    speaker_id=None,
                    bbox=bbox,
                    raw_ref=f"ocr.json#{item_id}"
                )
                records.append(record)
        except ValidationError as ve:
            logger.warning(f"Validation error for OCR item {item.get('id')}: {ve}")
        except Exception as e:
            logger.warning(f"Unexpected error processing OCR item: {e}")
            
    return records

def parse_keywords(file_path: str, video_id: str) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    data = _load_json_safe(file_path)
    
    for item in data:
        try:
            item_id = item.get("id", 0)
            text = item.get("text", "")
            confidence = float(item.get("confidence", 0.0))
            instances = item.get("instances", [])
            
            for idx, inst in enumerate(instances):
                start_ts = inst.get("start", "")
                end_ts = inst.get("end", "")
                
                record = EvidenceRecord(
                    evidence_id=f"{video_id}keyword{item_id}_{idx}",
                    video_id=video_id,
                    source_type="keyword",
                    source_item_id=item_id,
                    text=text,
                    normalized_text=normalize_text(text),
                    confidence=confidence,
                    start_time_ms=parse_timestamp(start_ts),
                    end_time_ms=parse_timestamp(end_ts),
                    speaker_id=None,
                    bbox=None,
                    raw_ref=f"keywords.json#{item_id}"
                )
                records.append(record)
        except ValidationError as ve:
            logger.warning(f"Validation error for keyword item {item.get('id')}: {ve}")
        except Exception as e:
            logger.warning(f"Unexpected error processing keyword item: {e}")
            
    return records

if __name__ == "__main__":
    import os
    import tempfile
    
    # Mock data definition
    mock_transcript = [
        {
            "id": 1,
            "text": "Welcome to the Azure, GraphRAG demo!",
            "confidence": 0.98,
            "speakerId": 2,
            "language": "en-US",
            "instances": [
                {"start": "0:00:01.500", "end": "0:00:04.250"},
                {"start": "0:01:05.100", "end": "0:01:08.000"}
            ]
        }
    ]
    
    mock_ocr = [
        {
            "id": 12,
            "text": "SYSTEM ARCHITECTURE",
            "confidence": 0.95,
            "left": 100,
            "top": 50,
            "width": 800,
            "height": 120,
            "angle": 0,
            "language": "en-US",
            "instances": [
                {"start": "0:00:10.000", "end": "0:00:15.500"}
            ]
        }
    ]
    
    mock_keywords = [
        {
            "id": 5,
            "text": "Machine Learning",
            "confidence": 0.88,
            "language": "en-US",
            "instances": [
                {"start": "0:05:00.000", "end": "0:05:10.000"}
            ]
        }
    ]
    
    # Create temporary directory to hold our mock inputs
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = os.path.join(tmpdir, "transcript.json")
        ocr_path = os.path.join(tmpdir, "ocr.json")
        keywords_path = os.path.join(tmpdir, "keywords.json")
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(mock_transcript, f)
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(mock_ocr, f)
        with open(keywords_path, "w", encoding="utf-8") as f:
            json.dump(mock_keywords, f)
            
        test_video_id = "v_sample_999"
        
        # Execute parsing logic
        all_records: List[EvidenceRecord] = []
        all_records.extend(parse_transcript(transcript_path, test_video_id))
        all_records.extend(parse_ocr(ocr_path, test_video_id))
        all_records.extend(parse_keywords(keywords_path, test_video_id))
        
        # Output results
        print(f"Total EvidenceRecords Generated: {len(all_records)}\n")
        for record in all_records:
            print(record.model_dump_json(indent=2))