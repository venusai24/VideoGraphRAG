import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AVIClient:
    def __init__(self, api_key: str, account_id: str, location: str = "trial", account_type: str = "trial"):
        self.api_key = api_key
        self.account_id = account_id
        self.location = "trial" if account_type.lower() == "trial" else location
        self.base_url = "https://api.videoindexer.ai"
        self.access_token = None

    def _get_headers(self):
        if not self.access_token:
            self.get_fresh_access_token()
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_fresh_access_token(self):
        logger.info("Attempting to fetch fresh access token via Subscription Key...")
        url = f"{self.base_url}/auth/{self.location}/Accounts/{self.account_id}/AccessToken?allowEdit=true"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            self.access_token = response.text.strip('"')
            return self.access_token
        else:
            logger.error(f"Token Request Failed: {response.status_code} - {response.text}")
            raise Exception(f"Failed to get access token: {response.text}")

    def upload_video(self, video_path: str, video_name: str) -> str:
        url = f"{self.base_url}/{self.location}/Accounts/{self.account_id}/Videos"
        params = {
            "name": video_name,
            "privacy": "Private",
            "indexingPreset": "Default",
            "language": "en-US"
        }
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Local video not found: {video_path}")
            
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, headers=self._get_headers(), params=params, files=files)
            
        if response.status_code == 401:
            self.get_fresh_access_token()
            with open(video_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, headers=self._get_headers(), params=params, files=files)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Upload Failed: {response.status_code} - {response.text}")
            
        video_id = response.json().get("id")
        logger.info(f"Upload Successful. Target Video ID: {video_id}")
        return video_id

    def wait_for_index(self, video_id: str, timeout: int = 1800) -> Dict[str, Any]:
        url = f"{self.base_url}/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(url, headers=self._get_headers())
            if response.status_code == 401:
                self.get_fresh_access_token()
                response = requests.get(url, headers=self._get_headers())
                
            response.raise_for_status()
            data = response.json()
            
            state = data.get("state")
            if not state and "videos" in data:
                state = data["videos"][0].get("state")
                
            logger.info(f"Polling Status: {state} ({int(time.time() - start_time)}s elapsed)")
            
            if state == "Processed":
                return data
            elif state == "Failed":
                failure_message = data.get("videos", [{}])[0].get("failureMessage", str(data))
                raise Exception(f"Azure Video Indexer failure: {failure_message}")
                
            time.sleep(30)
            
        raise TimeoutError("Indexing timed out.")

    def extract_structured_data(self, full_insight: Dict[str, Any]) -> Dict[str, Any]:
        video = full_insight.get("videos", [{}])[0]
        insights = video.get("insights", {})
        
        return {
            "transcript": insights.get("transcript", []),
            "ocr": insights.get("ocr", []),
            "scenes": insights.get("scenes", []),
            "shots": insights.get("shots", []),
            "keywords": insights.get("keywords", []),
            "topics": insights.get("topics", [])
        }

    def generate_rag_chunks(self, video_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        transcript = data.get("transcript", [])
        chunks = []
        
        if not transcript:
            return []

        current_chunk_text = ""
        chunk_start = transcript[0].get("instances", [{}])[0].get("start", "00:00:00")
        MAX_CHAR_LENGTH = 1500
        keywords = [k.get("text") for k in data.get("keywords", [])[:5]]

        for i, entry in enumerate(transcript):
            text = entry.get("text", "")
            instances = entry.get("instances", [{}])[0]
            end_time = instances.get("end", "00:00:00")
            
            if len(current_chunk_text) + len(text) > MAX_CHAR_LENGTH:
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
                
        if current_chunk_text:
            chunks.append({
                "text": current_chunk_text.strip(),
                "start_time": chunk_start,
                "end_time": transcript[-1].get("instances", [{}])[0].get("end", "00:00:00"),
                "video_id": video_id,
                "keywords": keywords,
                "source": "azure_video_indexer"
            })
            
        return chunks
