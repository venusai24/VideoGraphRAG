import os
import time
import json
import logging
import requests
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Set up specialized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('avi_project_creation.log')
    ]
)
logger = logging.getLogger(__name__)

class AzureVideoIndexerProjectBuilder:
    def __init__(self):
        self.config = {}
        self.base_url = "https://api.videoindexer.ai"
        self.base_output_dir = "outputs/projects"

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
            
            if "TOKEN" in var:
                self.config[var] = val
                logger.info(f"Loaded {var}: {val[:6]}...{val[-6:]}")
            else:
                self.config[var] = val
                logger.info(f"Loaded {var}: {val}")

    def _get_headers(self, token: str = None):
        t = token or self.config.get("VIDEO_INDEXER_ACCESS_TOKEN")
        return {
            "Authorization": f"Bearer {t}"
        }

    def _get_location(self):
        return "trial" if self.config["VIDEO_INDEXER_ACCOUNT_TYPE"].lower() == "trial" else self.config["VIDEO_INDEXER_LOCATION"]

    def get_fresh_access_token(self):
        """Task 1.1: Fetch fresh access token using API Key"""
        api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
        if not api_key:
            return None
            
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
        except Exception as e:
            logger.error(f"Network error during token fetch: {str(e)}")
        return None

    def create_project(self, video_details: List[Dict[str, str]]) -> str:
        """Task 5: Create a Project using the EXISTING indexed videos"""
        logger.info("Initializing Task 5: Creating Project with updated POST JSON payload")
        
        # Ensure token is valid before creating project
        self.get_fresh_access_token() 
        
        loc = self._get_location()
        
        # FIX: Switched to POST and removed {projectId} from the URL to CREATE a new project
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Projects"
        
        project_name = f"Aggregated-Project-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Maintain the updated 'videosRanges' schema
        videos_ranges = []
        for vid in video_details:
            videos_ranges.append({
                "videoId": vid["id"],
                "range": {
                    "start": "0:00:00.000",
                    "end": vid["duration"]
                }
            })

        payload = {
            "name": project_name,
            "videosRanges": videos_ranges,
            "isSearchable": None 
        }
        
        # Execute the POST request
        response = requests.post(url, headers=self._get_headers(), json=payload)
        
        if response.status_code not in [200, 201]:
            logger.error(f"Project Creation Failed: {response.text}")
            response.raise_for_status()
            
        # Extract the server-generated project ID from the response
        project_id = response.json().get("id")
        logger.info(f"Project created successfully! Project ID: {project_id}")
        return project_id

    def get_project_index(self, project_id: str) -> Dict[str, Any]:
        """Task 6: Retrieve Final Project Index"""
        logger.info(f"Initializing Task 6: Fetching Project Index for {project_id}")
        loc = self._get_location()
        url = f"{self.base_url}/{loc}/Accounts/{self.config['VIDEO_INDEXER_ACCOUNT_ID']}/Projects/{project_id}/Index"
        
        params = {
            "includeSummarizedInsights": "false"
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code != 200:
            logger.error(f"Failed to fetch Project Index: {response.text}")
            response.raise_for_status()
            
        logger.info("Successfully fetched Project Index.")
        return response.json()

    def save_project_output(self, project_id: str, data: Dict[str, Any]):
        """Task 7: Store Project Output in JSON"""
        logger.info("Initializing Task 7: Persisting Project JSON")
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        output_path = os.path.join(self.base_output_dir, f"project_{project_id}_index.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Project insights stored successfully at: {output_path}")

    def run_creation_only(self):
        logger.info("=== STARTING AZURE PROJECT ASSEMBLY (SKIPPING UPLOADS) ===")
        try:
            self.validate_config()
            
            processed_video_details = [
                {"id": "e4h7j816m7", "duration": "0:00:30.102"},
                {"id": "v9krny4792", "duration": "0:00:30.049"},
                {"id": "xaoj2tjsd8", "duration": "0:00:30.079"},
                {"id": "1p8mdifbcb", "duration": "0:00:30.067"},
                {"id": "hrq3z2iqe8", "duration": "0:00:30.055"},
                {"id": "x44chuv1nj", "duration": "0:00:30.044"},
                {"id": "an7gepv28c", "duration": "0:00:30.074"},
                {"id": "4bc77dcwwd", "duration": "0:00:30.104"},
                {"id": "3a903xr8or", "duration": "0:00:14.835"}
            ]
            
            # Step 1: Combine existing videos into a project
            project_id = self.create_project(processed_video_details)
            
            # Step 2: Get and store the final Project Index
            project_data = self.get_project_index(project_id)
            self.save_project_output(project_id, project_data)
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Project ID: {project_id}")
            logger.info(f"Total Videos Indexed in Project: {len(processed_video_details)}")
            
        except Exception as e:
            logger.error(f"PIPELINE FAILED: {str(e)}")
            raise

if __name__ == "__main__":
    builder = AzureVideoIndexerProjectBuilder()
    builder.run_creation_only()