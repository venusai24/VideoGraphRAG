import torch
import time
import psutil
import json
import os
from transformers import pipeline
import numpy as np
import librosa

def get_peak_memory():
    # Only tracking RAM for now as MPS VRAM is unified and harder to isolate per process
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def benchmark_config(model_id, audio_path, device, batch_size, chunk_length_s):
    print(f"Testing: device={device}, batch={batch_size}, chunk={chunk_length_s}s")
    
    start_mem = get_peak_memory()
    
    # Force float16 for MPS to avoid common errors and gain speed
    torch_dtype = torch.float16 if device == "mps" else torch.float32
    
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=chunk_length_s
        )
        
        # Load audio once
        audio, sr = librosa.load(audio_path, sr=16000)
        # Repeat to simulate longer audio (~60s) for better throughput measurement
        audio_60s = np.tile(audio, int(60 / (len(audio) / sr)))
        
        start_time = time.time()
        result = pipe(audio_60s, batch_size=batch_size)
        end_time = time.time()
        
        peak_mem = get_peak_memory()
        duration = end_time - start_time
        audio_duration = len(audio_60s) / sr
        throughput = audio_duration / duration
        
        return {
            "status": "success",
            "device": device,
            "batch_size": batch_size,
            "chunk_length_s": chunk_length_s,
            "time_sec": round(duration, 2),
            "audio_duration_sec": round(audio_duration, 2),
            "throughput_ratio": round(throughput, 2),
            "peak_mem_mb": round(peak_mem, 2),
            "mem_growth_mb": round(peak_mem - start_mem, 2),
            "text_len": len(result["text"])
        }
    except Exception as e:
        return {
            "status": "error",
            "device": device,
            "batch_size": batch_size,
            "chunk_length_s": chunk_length_s,
            "error": str(e)
        }

if __name__ == "__main__":
    MODEL_ID = "openai/whisper-large-v3"
    AUDIO_PATH = "outputs/test_audio.wav"
    OUTPUT_FILE = "outputs/asr_benchmarks.json"
    
    # librosa might need to be installed
    # ./vgent/bin/pip install librosa
    
    configs = [
        {"device": "mps", "batch_size": 1, "chunk_length_s": 30},
        {"device": "mps", "batch_size": 2, "chunk_length_s": 30},
        {"device": "mps", "batch_size": 4, "chunk_length_s": 30},
        {"device": "mps", "batch_size": 1, "chunk_length_s": 10},
        {"device": "cpu", "batch_size": 1, "chunk_length_s": 30},
    ]
    
    results = []
    for cfg in configs:
        res = benchmark_config(MODEL_ID, AUDIO_PATH, **cfg)
        results.append(res)
        print(f"Result: {res.get('throughput_ratio', 'N/A')}x RT speed")
        
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {OUTPUT_FILE}")
