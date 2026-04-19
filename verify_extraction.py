import json
import os
from pathlib import Path

def audit_enrichment(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"\n--- AUDITING {path.name} ---")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    total = len(results)
    print(f"Total clips processed: {total}")

    if "vision" in str(path.name):
        entities_count = 0
        actions_count = 0
        total_confidence = 0.0
        confidence_items = 0
        categories = set()

        for res in results:
            vision = res.get("vision")
            if not vision:
                continue
            
            entities = vision.get("entities", [])
            actions = vision.get("actions", [])
            entities_count += len(entities)
            actions_count += len(actions)

            for ent in entities:
                categories.add(ent.get("category"))
                conf = ent.get("confidence")
                if conf is not None:
                    total_confidence += conf
                    confidence_items += 1

        avg_conf = total_confidence / confidence_items if confidence_items > 0 else 0
        print(f"Total Entities: {entities_count} (Avg {entities_count/total:.1f} per clip)")
        print(f"Total Actions: {actions_count} (Avg {actions_count/total:.1f} per clip)")
        print(f"Average Confidence: {avg_conf:.2f}")
        print(f"Categories found: {', '.join(filter(None, categories))}")
    
    elif "audio" in str(path.name):
        transcripts = 0
        empty = 0
        for res in results:
            audio = res.get("audio")
            if audio and audio.get("text"):
                transcripts += 1
            else:
                empty += 1
        print(f"Successful transcripts: {transcripts}")
        print(f"Empty/Failed: {empty}")

if __name__ == "__main__":
    audit_enrichment("outputs/enrichment_vision.jsonl")
    audit_enrichment("outputs/enrichment_audio.jsonl")
