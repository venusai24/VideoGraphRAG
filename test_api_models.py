import os
from dotenv import load_dotenv
import openai

load_dotenv()
c_key = os.environ.get("CEREBRAS_API_KEYS", "").split(",")[0]
g_key = os.environ.get("GROQ_API_KEYS", "").split(",")[0]

print("Testing Cerebras...")
try:
    c_client = openai.OpenAI(api_key=c_key, base_url="https://api.cerebras.ai/v1")
    for m in c_client.models.list().data:
        print("  Cerebras model:", m.id)
except Exception as e:
    print(e)

print("Testing Groq...")
try:
    g_client = openai.OpenAI(api_key=g_key, base_url="https://api.groq.com/openai/v1")
    for m in g_client.models.list().data:
        if 'qwen' in m.id.lower() or 'llama' in m.id.lower():
            print("  Groq model:", m.id)
except Exception as e:
    print(e)
