#!/bin/bash
# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "--- Testing Azure Video Indexer ---"
# Step 1: Get Access Token
TOKEN=$(curl -s -X GET "https://api.videoindexer.ai/auth/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/AccessToken?allowEdit=true" \
    -H "Ocp-Apim-Subscription-Key: ${AZURE_VIDEO_INDEXER_API_KEY}")

if [[ $TOKEN == *"ErrorCode"* ]]; then
    echo "Azure Auth FAILED: $TOKEN"
else
    echo "Azure Auth SUCCESS: Token acquired (starts with ${TOKEN:0:10}...)"
fi

echo -e "\n--- Testing OpenAI API ---"
curl -s -X POST "https://api.openai.com/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5
      }'

echo -e "\n--- Testing Leonardo.ai API ---"
# Testing a simple user info or usage endpoint
curl -s -X GET "https://cloud.leonardo.ai/api/rest/v1/me" \
    -H "accept: application/json" \
    -H "authorization: Bearer ${LEONARDO_API_KEY}"
