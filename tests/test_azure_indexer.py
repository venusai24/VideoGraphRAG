import os
import time
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AzureVideoIndexer:
    def __init__(self):
        self.api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
        self.account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
        self.location = os.getenv("AZURE_VIDEO_INDEXER_LOCATION", "trial")
        self.base_url = "https://api.videoindexer.ai"
        
        if not self.api_key or not self.account_id:
            raise ValueError("Missing AZURE_VIDEO_INDEXER_API_KEY or AZURE_VIDEO_INDEXER_ACCOUNT_ID in .env")

    def get_access_token(self):
        """Fetches an access token for the Video Indexer API."""
        print(f"--- Fetching Access Token for Account: {self.account_id} ---")
        url = f"{self.base_url}/auth/{self.location}/Accounts/{self.account_id}/AccessToken?allowEdit=true"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text.strip('"')
        except Exception as e:
            print(f"Auth Error: {e}")
            if response.text:
                print(f"Response Detail: {response.text}")
            raise

    def upload_video(self, video_path, video_name="TestVideo"):
        """Uploads a local video file to Azure Video Indexer."""
        token = self.get_access_token()
        print(f"--- Uploading Video: {video_path} ---")
        
        url = f"{self.base_url}/{self.location}/Accounts/{self.account_id}/Videos"
        params = {
            "name": video_name,
            "privacy": "Private",
            "accessToken": token,
            "indexingPreset": "AdvancedVideo",
            "streamingPreset": "Default"
        }
        
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        with open(video_path, 'rb') as video_file:
            files = {'file': video_file}
            response = requests.post(url, params=params, files=files)
            
        response.raise_for_status()
        video_data = response.json()
        print(f"Successfully started indexing. Video ID: {video_data['id']}")
        return video_data['id']

    def wait_for_index(self, video_id):
        """Polls the video status until indexing is complete."""
        url = f"{self.base_url}/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"
        
        print(f"--- Processing Video ID: {video_id} ---")
        while True:
            # Need fresh token periodically for long jobs
            token = self.get_access_token()
            params = {"accessToken": token}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # The state can be in the top level or inside the 'videos' array
            state = data.get("state")
            if not state and "videos" in data:
                state = data["videos"][0].get("state")
                
            print(f"Current State: {state}")
            
            if state == "Processed":
                return data
            elif state == "Failed":
                raise Exception(f"Video indexing failed: {data}")
            
            time.sleep(30) # Poll every 30 seconds

    def save_insights(self, data, output_path="outputs/avi_results.json"):
        """Saves the rich insights JSON to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"--- Insights saved to: {output_path} ---")

if __name__ == "__main__":
    # Change this to use one of your local files
    TARGET_VIDEO = "testClip.mp4" 
    
    indexer = AzureVideoIndexer()
    
    try:
        # Step 1: Upload
        video_id = indexer.upload_video(TARGET_VIDEO)
        
        # Step 2: Wait & Retrieve
        insights = indexer.wait_for_index(video_id)
        
        # Step 3: Save Result
        indexer.save_insights(insights)
        
        print("\nTest rig complete! You can now inspect 'outputs/avi_results.json'")
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
