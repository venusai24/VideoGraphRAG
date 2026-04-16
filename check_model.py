import torch
from transformers import pipeline
import sys

try:
    print("Attempting to load CohereLabs/cohere-transcribe-03-2026...")
    # Using device=mps if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model="CohereLabs/cohere-transcribe-03-2026", device=device)
    print("SUCCESS: Model loaded")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
