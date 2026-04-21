import os
import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Set up specialized logging for production validation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('avi_validation.log')
    ]
)
logger = logging.getLogger(__name__)

class AVIVersion:
    PROD_VALIDATION = "1.0.0"

class AzureVideoIndexerValidator:
    def __init__(self):
        self.config = {}
        self.base_url = "https://api.videoindexer.ai"
        self.output_dir = "outputs/video_indexer_test"
        
    def validate_config(self):
        """Task 1: Validate Configuration"""
        logger.info("Initializing Task 1: Configuration Validation")
        load_dotenv()
        
        required_vars = [
            "VIDEO_INDEXER_ACCESS_TOKEN",
            "VIDEO_INDEXER_ACCOUNT_ID",
            "VIDEO_INDEXER_LOCATION",
            "VIDEO_INDEXER_ACCOUNT_TYPE"
        ]
        
        for var in required_vars:
            val = os.getenv(var)
            if not val:
                logger.error(f"Missing mandatory environment variable: {var}")
                raise ValueError(f"CRITICAL: {var} is not defined in .env")
            
            # Sanitized logging
            if "TOKEN" in var:
                masked = f"{val[:6]}...{val[-6:]}"
                self.config[var] = val
                logger.info(f"Loaded {var}: {masked}")
            else:
                self.config[var] = val
                logger.info(f"Loaded {var}: {val}")

        # Handle TEST_VIDEO_URL (using the local file requested by the user)
        self.config["TEST_VIDEO_URL"] = os.getenv("TEST_VIDEO_URL", "when i breeth.mp4")
        logger.info(f"Test Video Path: {self.config['TEST_VIDEO_URL']}")

    def _get_headers(self, token: str = None):
        t = token or self.config.get("VIDEO_INDEXER_ACCESS_TOKEN")
        return {
            "Authorization": f"Bearer {t}"
        }

    def get_fresh_access_token(self):
        """Task 1.1: Fetch fresh access token using API Key (Subscription Key)"""
        api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
        if not api_key:
            logger.warning("AZURE_VIDEO_INDEXER_API_KEY not found in .env")
            return None
        
        # Diagnostic: JWT check
        if api_key.startswith("eyJ"):
            logger.error("CRITICAL: AZURE_VIDEO_INDEXER_API_KEY appears to be a JWT, not a Subscription Key.")
            logger.error("Please use the 32-character hex Primary Key from your VI Profile/API Access page.")
            return None
        
        logger.info("Attempting to fetch fresh access token via Subscription Key...")
        loc = "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]
        url = f"{self.base_url}/auth/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/AccessToken?allowEdit=true"
        
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                token = response.text.strip('"')
                self.config["VIDEO_INDEXER_ACCESS_TOKEN"] = token
                logger.info("Success: Fresh Access Token acquired.")
                return token
            else:
                logger.error(f"Token Request Failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Network error during token fetch: {str(e)}")
            return None

    def validate_connectivity(self):
        """Task 2: Validate API Connectivity"""
        logger.info("Initializing Task 2: API Connectivity Check")
        # Try current token
        loc = "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Videos?pageSize=1"
        
        response = requests.get(url, headers=self._get_headers())
        
        if response.status_code == 401:
            logger.warning("Token expired. Attempting rotation...")
            new_token = self.get_fresh_access_token()
            if new_token:
                response = requests.get(url, headers=self._get_headers(new_token))
        
        if response.status_code == 200:
            logger.info("Connectivity Validated: Status 200 OK")
        else:
            logger.error(f"Connectivity Failed: {response.status_code} - {response.text}")
            raise Exception(f"API Connectivity Error: {response.status_code}")

    def upload_test_video(self) -> str:
        """Task 3: Upload Test Video"""
        logger.info("Initializing Task 3: Uploading Test Video")
        loc = "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]
        
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Videos"
        
        test_video = self.config["TEST_VIDEO_URL"]
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_name = f"project-hail-mary-test-{timestamp}"
        
        params = {
            "name": video_name,
            "privacy": "Private",
            "indexingPreset": "AdvancedVideo",
            "language": "en-US"
        }
        
        response = None
        if test_video.startswith("http"):
            params["videoUrl"] = test_video
            response = requests.post(url, headers=self._get_headers(), params=params)
        else:
            if not os.path.exists(test_video):
                raise FileNotFoundError(f"Local test video not found: {test_video}")
            with open(test_video, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, headers=self._get_headers(), params=params, files=files)
        
        if response.status_code not in [200, 201]:
            logger.error(f"Upload Failed: {response.status_code} - {response.text}")
            response.raise_for_status()
            
        video_id = response.json().get("id")
        logger.info(f"Upload Successful. Target Video ID: {video_id}")
        return video_id

    def poll_status(self, video_id: str):
        """Task 4: Poll Indexing Status"""
        logger.info(f"Initializing Task 4: Polling status for {video_id}")
        loc = "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Videos/{video_id}/Index"
        
        start_time = time.time()
        timeout = 600 # 10 minutes
        
        while time.time() - start_time < timeout:
            response = requests.get(url, headers=self._get_headers())
            data = response.json()
            
            # Handle possible nested structure
            state = data.get("state")
            if not state and "videos" in data:
                state = data["videos"][0].get("state")
            
            logger.info(f"Polling Status: {state} ({int(time.time() - start_time)}s elapsed)")
            
            if state == "Processed":
                logger.info("Indexing Complete.")
                return data
            elif state == "Failed":
                failure_message = data.get("videos", [{}])[0].get("failureMessage", str(data))
                logger.error(f"Indexing Failed. Reason: {failure_message}")
                raise Exception(f"Azure Video Indexer reported a failure state: {failure_message}")
            
            time.sleep(20)
            
        raise TimeoutError("Indexing timed out after 10 minutes.")

    def timestamp_to_seconds(self, ts: str) -> float:
        """Helper: Convert HH:MM:SS.mmm to seconds"""
        if not ts: return 0.0
        try:
            # Handle "0:00:05.123" or similar formats
            parts = ts.split(':')
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s
        except:
            return 0.0

    def extract_structured_data(self, full_insight: Dict[str, Any]) -> Dict[str, Any]:
        """Task 6: Extract Structured Data"""
        logger.info("Initializing Task 6: Data Extraction")
        video = full_insight.get("videos", [{}])[0]
        insights = video.get("insights", {})
        
        extracted = {
            "transcript": insights.get("transcript", []),
            "ocr": insights.get("ocr", []),
            "scenes": insights.get("scenes", []),
            "shots": insights.get("shots", []),
            "keywords": insights.get("keywords", []),
            "topics": insights.get("topics", [])
        }
        
        logger.info(f"Extracted {len(extracted['transcript'])} transcript segments")
        logger.info(f"Extracted {len(extracted['ocr'])} OCR elements")
        return extracted

    def generate_rag_chunks(self, video_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Task 7: Transform into RAG-ready format"""
        logger.info("Initializing Task 7: RAG Chunking")
        transcript = data.get("transcript", [])
        chunks = []
        
        if not transcript:
            return []

        current_chunk_text = ""
        chunk_start = transcript[0].get("instances", [{}])[0].get("start", "00:00:00")
        
        # Token estimation (rough: 1 token ~ 4 chars)
        MAX_CHAR_LENGTH = 1500 # Approx 375 tokens
        
        keywords = [k.get("text") for k in data.get("keywords", [])[:5]]

        for i, entry in enumerate(transcript):
            text = entry.get("text", "")
            instances = entry.get("instances", [{}])[0]
            end_time = instances.get("end", "00:00:00")
            
            if len(current_chunk_text) + len(text) > MAX_CHAR_LENGTH:
                # Close current chunk
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "start_time": chunk_start,
                    "end_time": instances.get("start", end_time),
                    "video_id": video_id,
                    "keywords": keywords,
                    "source": "azure_video_indexer"
                })
                current_chunk_text = text
                chunk_start = instances.get("start", "00:00:00")
            else:
                current_chunk_text += " " + text
        
        # Add final chunk
        if current_chunk_text:
            chunks.append({
                "text": current_chunk_text.strip(),
                "start_time": chunk_start,
                "end_time": transcript[-1].get("instances", [{}])[0].get("end", "00:00:00"),
                "video_id": video_id,
                "keywords": keywords,
                "source": "azure_video_indexer"
            })

        logger.info(f"Generated {len(chunks)} RAG chunks")
        return chunks

    def delete_video(self, video_id: str):
        """Task 10: Delete Video from Azure Video Indexer"""
        logger.info(f"Initializing Task 10: Deleting video {video_id}")
        loc = "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Videos/{video_id}"
        
        response = requests.delete(url, headers=self._get_headers())
        
        if response.status_code in [200, 204]:
            logger.info(f"Successfully deleted video {video_id} from storage.")
        else:
            logger.error(f"Failed to delete video: {response.status_code} - {response.text}")

    def save_outputs(self, raw_data: Dict[str, Any], extracted: Dict[str, Any], chunks: List[Dict[str, Any]]):
        """Task 8: Persist to File System"""
        logger.info(f"Initializing Task 8: Saving artifacts to {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        files_to_save = {
            "raw_insights.json": raw_data,
            "transcript.json": extracted["transcript"],
            "ocr.json": extracted["ocr"],
            "scenes.json": extracted["scenes"],
            "keywords.json": extracted["keywords"],
            "rag_chunks.json": chunks
        }
        
        for name, content in files_to_save.items():
            path = os.path.join(self.output_dir, name)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        
        logger.info("All artifacts persisted successfully.")

    def run_validation(self):
        logger.info("=== STARTING AZURE VIDEO INDEXER PRODUCTION VALIDATION ===")
        try:
            self.validate_config()
            self.validate_connectivity()
            
            video_id = self.upload_test_video()
            full_insight = self.poll_status(video_id)
            
            extracted = self.extract_structured_data(full_insight)
            chunks = self.generate_rag_chunks(video_id, extracted)
            
            self.save_outputs(full_insight, extracted, chunks)
            
            # Final Validation (Task 9)
            logger.info("=== FINAL VALIDATION RESULTS ===")
            logger.info(f"Video ID: {video_id}")
            logger.info(f"Total Transcript Segments: {len(extracted['transcript'])}")
            logger.info(f"Total RAG Chunks: {len(chunks)}")
            
            if not extracted['transcript']:
                logger.warning("VALIDATION WARNING: Transcript is empty!")
            else:
                logger.info("VALIDATION SUCCESS: Transcript verified.")
                
            if len(chunks) > 0:
                logger.info("VALIDATION SUCCESS: RAG Chunks generated.")
            
            # Step 10: Delete
            self.delete_video(video_id)
            
        except Exception as e:
            logger.error(f"VALIDATION FAILED: {str(e)}")
            raise

if __name__ == "__main__":
    validator = AzureVideoIndexerValidator()
    validator.run_validation()
