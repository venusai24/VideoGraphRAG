import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_azure():
    print("--- Testing Azure Video Indexer ---")
    api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
    account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
    location = os.getenv("AZURE_VIDEO_INDEXER_LOCATION", "trial")
    
    url = f"https://api.videoindexer.ai/auth/{location}/Accounts/{account_id}/AccessToken?allowEdit=true"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            token = response.text.strip('"')
            print(f"SUCCESS: Azure Token acquired ({token[:10]}...)")
            return True
        else:
            print(f"FAILED: Azure Auth {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"ERROR: Azure test failed: {e}")
        return False

def test_openai():
    print("\n--- Testing OpenAI ---")
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print("SUCCESS: OpenAI API is working.")
            return True
        else:
            print(f"FAILED: OpenAI {response.status_code} - {response.json().get('error', {}).get('message', response.text)}")
            return False
    except Exception as e:
        print(f"ERROR: OpenAI test failed: {e}")
        return False

def test_leonardo():
    print("\n--- Testing Leonardo.ai ---")
    api_key = os.getenv("LEONARDO_API_KEY")
    # Using a common GET endpoint
    url = "https://cloud.leonardo.ai/api/rest/v1/me"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("SUCCESS: Leonardo API is working.")
            return True
        else:
            print(f"FAILED: Leonardo {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"ERROR: Leonardo test failed: {e}")
        return False

if __name__ == "__main__":
    test_azure()
    test_openai()
    test_leonardo()
