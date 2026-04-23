import os
import time
import json
import logging
import requests
import concurrent.futures
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Set up specialized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('avi_project_validation.log')
    ]
)
logger = logging.getLogger(__name__)

class AzureVideoIndexerProjectManager:
    def __init__(self):
        self.config = {}
        self.base_url = "https://api.videoindexer.ai"
        self.base_output_dir = "outputs/projects"

    def validate_config(self):
        """Task 1: Validate Configuration and Local Files"""
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

        # ---------------------------------------------------------
        # THIS IS WHERE YOUR CLIPS ARE DEFINED
        # Generates: ['clip_000.mp4', 'clip_001.mp4' ... 'clip_008.mp4']
        # ---------------------------------------------------------
        self.config["LOCAL_VIDEO_FILES"] = [f"clip_{str(i).zfill(3)}.mp4" for i in range(9)]
        
        # Verify the files actually exist in your directory before starting
        for file_path in self.config["LOCAL_VIDEO_FILES"]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CRITICAL: Local video file not found: {file_path}. Please place it in the same folder as this script.")
                
        logger.info(f"Loaded and verified {len(self.config['LOCAL_VIDEO_FILES'])} local videos for the project.")

    def _get_headers(self, token: str = None):
        t = token or self.config.get("VIDEO_INDEXER_ACCESS_TOKEN")
        return {
            "Authorization": f"Bearer {t}"
        }

    def _get_location(self):
        return "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]

    def get_fresh_access_token(self):
        """Task 1.1: Fetch fresh access token using API Key (Subscription Key)"""
        api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
        if not api_key:
            logger.warning("AZURE_VIDEO_INDEXER_API_KEY not found in .env")
            return None
            
        logger.info("Attempting to fetch fresh access token via Subscription Key...")
        loc = self._get_location()
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
        loc = self._get_location()
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

    def _upload_single_video(self, file_path: str, index: int) -> str:
        """Helper for uploading a single local video file"""
        loc = self._get_location()
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Videos"
        
        video_name = os.path.basename(file_path)
        
        params = {
            "name": video_name,
            "privacy": "Private",
            "indexingPreset": "Default",
            "language": "en-US"
        }
        
        logger.info(f"Uploading local video {index}: {video_name}...")
        
        # Open the file and send it as multipart/form-data
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, headers=self._get_headers(), params=params, files=files)
            
        if response.status_code not in [200, 201]:
            logger.error(f"Upload Failed for {video_name}: {response.text}")
            response.raise_for_status()
        
        video_id = response.json().get("id")
        logger.info(f"Video {index} ({video_name}) uploaded successfully. Target ID: {video_id}")
        return video_id

    def upload_videos_parallelly(self) -> List[str]:
        """Task 3: Upload Local Videos Parallelly"""
        logger.info("Initializing Task 3: Parallel Local Video Uploads")
        local_files = self.config["LOCAL_VIDEO_FILES"]
        video_ids = []
        
        # Limit max_workers to 5 to avoid overwhelming local network bandwidth
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(local_files))) as executor:
            future_to_file = {executor.submit(self._upload_single_video, file_path, i): file_path for i, file_path in enumerate(local_files)}
            
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    video_id = future.result()
                    video_ids.append(video_id)
                except Exception as exc:
                    logger.error(f"A video upload failed: {exc}")
                    
        if not video_ids:
            raise RuntimeError("All video uploads failed. Cannot create project.")
            
        return video_ids

    def _poll_single_video(self, video_id: str) -> Optional[Dict[str, str]]:
        """Helper to poll a single video until 'Processed' and extract its duration"""
        loc = self._get_location()
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Videos/{video_id}/Index"
        start_time = time.time()
        timeout = 1800 # 30 minutes
        
        while time.time() - start_time < timeout:
            response = requests.get(url, headers=self._get_headers())
            data = response.json()
            state = data.get("state", "Unknown")
            
            if state == "Processed":
                # Extract the formatted duration (e.g., "0:01:25.333") from the index payload
                duration = data.get("duration", "0:00:00.000") 
                if duration == "0:00:00.000" and "videos" in data:
                     duration = data["videos"][0].get("duration", "0:00:00.000")

                logger.info(f"Video {video_id} indexing Complete. Duration: {duration}")
                return {"id": video_id, "duration": duration}
                
            elif state == "Failed":
                logger.error(f"Video {video_id} indexing Failed.")
                return None
                
            time.sleep(15) # Wait before checking again
            
        logger.error(f"Polling timed out for video {video_id}")
        return None

    def wait_for_all_videos(self, video_ids: List[str]) -> List[Dict[str, str]]:
        """Task 4: Poll all videos parallelly and return their details"""
        logger.info(f"Initializing Task 4: Polling status for {len(video_ids)} videos")
        
        completed_videos = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(video_ids)) as executor:
            results = list(executor.map(self._poll_single_video, video_ids))
            
            for res in results:
                if res: # Filter out any that timed out or failed
                    completed_videos.append(res)
                    
        if len(completed_videos) < len(video_ids):
            logger.warning(f"Only {len(completed_videos)} out of {len(video_ids)} videos processed successfully.")
            
        return completed_videos

    def create_project(self, video_details: List[Dict[str, str]]) -> str:
        """Task 5: Create a Project using the indexed videos"""
        logger.info("Initializing Task 5: Creating Project with strict API schema")
        loc = self._get_location()
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Projects"
        
        project_name = f"Aggregated-Project-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        params = {"name": project_name}
        
        # Strictly matching the Azure Video Indexer Project Timeline schema
        instances = []
        for vid in video_details:
            instances.append({
                "videoInfo": {
                    "accountId": self.config["VIDEO_INDEXER_ACCOUNT_ID"],
                    "id": vid["id"]
                },
                "start": "0:00:00.000",
                "end": vid["duration"]
            })

        payload = {
            "timeLine": {
                "instances": instances
            }
        }
        
        response = requests.post(url, headers=self._get_headers(), params=params, json=payload)
        
        if response.status_code not in [200, 201]:
            logger.error(f"Project Creation Failed: {response.text}")
            response.raise_for_status()
            
        project_id = response.json().get("id")
        logger.info(f"Project created successfully! Project ID: {project_id}")
        return project_id

    def get_project_index(self, project_id: str) -> Dict[str, Any]:
        """Task 6: Retrieve Final Project Index"""
        logger.info(f"Initializing Task 6: Fetching Project Index for {project_id}")
        loc = self._get_location()
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Projects/{project_id}/Index"
        
        # Forcing includeSummarizedInsights to true retrieves the summarized blocks.
        params = {
            "includeSummarizedInsights": "true"
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code != 200:
            logger.error(f"Failed to fetch Project Index: {response.text}")
            response.raise_for_status()
            
        logger.info("Successfully fetched Project Index (including summarized insights).")
        return response.json()

    def save_project_output(self, project_id: str, data: Dict[str, Any]):
        """Task 7: Store Project Output in JSON"""
        logger.info("Initializing Task 7: Persisting Project JSON")
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        output_path = os.path.join(self.base_output_dir, f"project_{project_id}_index.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Project insights stored successfully at: {output_path}")

    def run_pipeline(self):
        logger.info("=== STARTING AZURE VIDEO INDEXER PROJECT PIPELINE ===")
        try:
            self.validate_config()
            self.validate_connectivity()
            
            # Step 1: Upload local files parallelly
            raw_video_ids = self.upload_videos_parallelly()
            
            # Step 2: Poll parallelly until processed
            processed_video_details = self.wait_for_all_videos(raw_video_ids)
            
            if not processed_video_details:
                raise Exception("No videos processed successfully. Cannot create project.")
            
            # Step 3: Combine into a project
            project_id = self.create_project(processed_video_details)
            
            # Step 4: Get and store the final Project Index
            project_data = self.get_project_index(project_id)
            self.save_project_output(project_id, project_data)
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Project ID: {project_id}")
            logger.info(f"Total Videos Indexed in Project: {len(processed_video_details)}")
            
        except Exception as e:
            logger.error(f"PIPELINE FAILED: {str(e)}")
            raise

if __name__ == "__main__":
    manager = AzureVideoIndexerProjectManager()
    manager.run_pipeline()