import os
import requests
from dotenv import load_dotenv

load_dotenv()

def delete_all_videos():
    api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
    account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
    location = os.getenv("AZURE_VIDEO_INDEXER_LOCATION", "trial")
    base_url = "https://api.videoindexer.ai"

    print(f"--- Purging all storage for Account: {account_id} ---")

    # Step 1: Get Access Token
    auth_url = f"{base_url}/auth/{location}/Accounts/{account_id}/AccessToken?allowEdit=true"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    try:
        token_response = requests.get(auth_url, headers=headers)
        token_response.raise_for_status()
        token = token_response.text.strip('"')
    except Exception as e:
        print(f"Auth Error: {e}")
        return

    # Step 2: List All Videos
    list_url = f"{base_url}/{location}/Accounts/{account_id}/Videos"
    params = {"accessToken": token}
    
    try:
        list_response = requests.get(list_url, params=params)
        list_response.raise_for_status()
        videos = list_response.json().get("results", [])
    except Exception as e:
        print(f"Error listing videos: {e}")
        return

    if not videos:
        print("No videos found in storage.")
        return

    print(f"Found {len(videos)} videos. Starting deletion...")

    # Step 3: Delete Each Video
    for video in videos:
        video_id = video.get("id")
        name = video.get("name")
        print(f"Deleting video: {name} (ID: {video_id})")
        
        delete_url = f"{base_url}/{location}/Accounts/{account_id}/Videos/{video_id}"
        try:
            # Re-fetch token if needed, but for small batches it should be fine
            del_response = requests.delete(delete_url, params={"accessToken": token})
            if del_response.status_code in [200, 204]:
                print(f"Successfully deleted {video_id}")
            else:
                print(f"Failed to delete {video_id}: {del_response.status_code}")
        except Exception as e:
            print(f"Error deleting {video_id}: {e}")

    print("--- Storage purge complete ---")

if __name__ == "__main__":
    delete_all_videos()
