import os
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.videoindexer.ai"


def validate_env():
    required_vars = [
        "AZURE_VIDEO_INDEXER_API_KEY",
        "VIDEO_INDEXER_ACCOUNT_ID",
        "VIDEO_INDEXER_LOCATION",
        "VIDEO_INDEXER_ACCOUNT_TYPE"
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")


def get_access_token():
    subscription_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
    account_id = os.getenv("VIDEO_INDEXER_ACCOUNT_ID")
    location = os.getenv("VIDEO_INDEXER_LOCATION")
    account_type = os.getenv("VIDEO_INDEXER_ACCOUNT_TYPE")

    loc = "trial" if account_type.lower() == "trial" else location

    url = f"{BASE_URL}/Auth/{loc}/Accounts/{account_id}/AccessToken"

    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }

    params = {
        "allowEdit": "true"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"❌ Failed to get access token: {response.text}")

    # token comes as string with quotes
    return response.text.strip('"')


def delete_video(video_id: str):
    account_id = os.getenv("VIDEO_INDEXER_ACCOUNT_ID")
    location = os.getenv("VIDEO_INDEXER_LOCATION")
    account_type = os.getenv("VIDEO_INDEXER_ACCOUNT_TYPE")

    loc = "trial" if account_type.lower() == "trial" else location

    url = f"{BASE_URL}/{loc}/Accounts/{account_id}/Videos/{video_id}"

    # 🔥 always fetch fresh token
    access_token = get_access_token()

    params = {
        "accessToken": access_token
    }

    print(f"🧹 Deleting video: {video_id}")

    response = requests.delete(url, params=params)

    # 🔁 Retry once if token expired
    if response.status_code == 401:
        print("🔁 Token expired, fetching new token and retrying...")
        access_token = get_access_token()
        params["accessToken"] = access_token
        response = requests.delete(url, params=params)

    if response.status_code in [200, 204]:
        print("✅ Video deleted successfully.")
    else:
        print(f"❌ Failed to delete video: {response.status_code}")
        print(response.text)
        response.raise_for_status()


if __name__ == "__main__":
    load_dotenv()
    validate_env()

    VIDEO_ID = "1haz6k8w8i"  # 👈 your video ID
    delete_video(VIDEO_ID)