# Deprecated – not used in final pipeline
"""
fetch_avi_results.py — Fetch Azure Video Indexer Results
=========================================================
For videos that were successfully indexed in AVI but whose JSON outputs
were never written to the output directory.

This script:
1. Lists all videos in your AVI account
2. Matches them by name pattern (video_<hash>) to output directories
3. Fetches full insights and extracts structured data
4. Writes all JSONs to outputs/<video_id>/

Usage:
    # Fetch for specific video IDs
    python fetch_avi_results.py 39fbc2c8 ac1682fb 303cbc17 43a38484

    # Fetch for specific video IDs using known AVI video IDs
    python fetch_avi_results.py --avi-id n2wq3edbkn --output-dir outputs/ac1682fb

    # List all videos in AVI account (discover video IDs)
    python fetch_avi_results.py --list

    # Fetch all videos that have output dirs but missing JSONs
    python fetch_avi_results.py --auto
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("fetch_avi.log", mode="a")])
logger = logging.getLogger("FetchAVI")

# ── Resolve imports ──────────────────────────────────────────────────
PREPROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_rag_preprocessing")
sys.path.insert(0, PREPROC_DIR)

from avi_client import AVIClient

load_dotenv()

EXPECTED_JSON_FILES = ["transcript.json", "ocr.json", "scenes.json", "keywords.json", "rag_chunks.json", "raw_insights.json"]


def create_avi_client() -> AVIClient:
    """Create an AVI client from environment variables."""
    return AVIClient(
        api_key=os.getenv("AZURE_VIDEO_INDEXER_API_KEY", ""),
        account_id=os.getenv("VIDEO_INDEXER_ACCOUNT_ID", ""),
        location=os.getenv("VIDEO_INDEXER_LOCATION", "trial"),
        account_type=os.getenv("VIDEO_INDEXER_ACCOUNT_TYPE", "trial"),
    )


def list_avi_videos(client: AVIClient) -> list:
    """List all videos in the AVI account."""
    import requests

    url = f"{client.base_url}/{client.location}/Accounts/{client.account_id}/Videos"
    headers = client._get_headers()

    response = requests.get(url, headers=headers)
    if response.status_code == 401:
        client.get_fresh_access_token()
        headers = client._get_headers()
        response = requests.get(url, headers=headers)

    response.raise_for_status()
    data = response.json()

    videos = data.get("results", [])
    return videos


def fetch_video_insights(client: AVIClient, avi_video_id: str) -> dict:
    """Fetch full insights for a specific AVI video ID."""
    import requests

    url = f"{client.base_url}/{client.location}/Accounts/{client.account_id}/Videos/{avi_video_id}/Index"
    headers = client._get_headers()

    response = requests.get(url, headers=headers)
    if response.status_code == 401:
        client.get_fresh_access_token()
        headers = client._get_headers()
        response = requests.get(url, headers=headers)

    response.raise_for_status()
    return response.json()


def save_avi_results(client: AVIClient, avi_video_id: str, output_dir: str, video_id_for_chunks: str):
    """Fetch insights from AVI and save all JSONs to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching insights for AVI video ID: {avi_video_id}...")
    full_insight = fetch_video_insights(client, avi_video_id)

    # Extract structured data
    extracted = client.extract_structured_data(full_insight)
    chunks = client.generate_rag_chunks(video_id_for_chunks, extracted)

    # Save all JSONs
    files_to_save = {
        "raw_insights.json": full_insight,
        "transcript.json": extracted["transcript"],
        "ocr.json": extracted["ocr"],
        "scenes.json": extracted["scenes"],
        "keywords.json": extracted["keywords"],
        "rag_chunks.json": chunks,
    }

    for filename, data in files_to_save.items():
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved: {filepath}")

    logger.info(f"  scenes={len(extracted.get('scenes', []))}, "
                f"transcript_segs={len(extracted.get('transcript', []))}, "
                f"ocr_items={len(extracted.get('ocr', []))}, "
                f"keywords={len(extracted.get('keywords', []))}, "
                f"rag_chunks={len(chunks)}")

    return True


def find_output_dirs_missing_jsons(outputs_root: str = "outputs") -> list:
    """Find output directories that have optimal_audio.wav but are missing JSON files."""
    root = Path(outputs_root)
    if not root.exists():
        return []

    missing = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        has_audio = (d / "optimal_audio.wav").exists()
        has_jsons = all((d / f).exists() for f in EXPECTED_JSON_FILES)

        if has_audio and not has_jsons:
            missing.append(str(d))

    return missing


def main():
    parser = argparse.ArgumentParser(description="Fetch AVI results for indexed videos")
    parser.add_argument("video_ids", nargs="*", help="Output folder video IDs (e.g. 39fbc2c8)")
    parser.add_argument("--list", action="store_true", help="List all videos in AVI account")
    parser.add_argument("--auto", action="store_true", help="Auto-discover and fetch for all missing")
    parser.add_argument("--avi-id", type=str, help="Specific AVI video ID to fetch")
    parser.add_argument("--output-dir", type=str, help="Output directory (used with --avi-id)")
    args = parser.parse_args()

    client = create_avi_client()

    if args.list:
        logger.info("Listing all videos in AVI account...")
        videos = list_avi_videos(client)
        logger.info(f"Found {len(videos)} videos:")
        for v in videos:
            state = v.get("state", "unknown")
            name = v.get("name", "?")
            vid = v.get("id", "?")
            duration = v.get("durationInSeconds", "?")
            logger.info(f"  {vid}  |  {name}  |  state={state}  |  duration={duration}s")
        return

    if args.avi_id and args.output_dir:
        # Direct fetch by AVI video ID
        video_id_for_chunks = Path(args.output_dir).name
        save_avi_results(client, args.avi_id, args.output_dir, video_id_for_chunks)
        return

    # Get all AVI videos for matching
    logger.info("Fetching AVI video list for name matching...")
    avi_videos = list_avi_videos(client)
    avi_name_map = {}  # local_video_id -> avi_video_id
    for v in avi_videos:
        name = v.get("name", "")
        avi_id = v.get("id", "")
        state = v.get("state", "")
        # The orchestrator names uploads as "video_<hash>"
        if name.startswith("video_"):
            local_id = name.replace("video_", "")
            if state == "Processed":
                avi_name_map[local_id] = avi_id
                logger.info(f"  Matched: {local_id} → {avi_id} (Processed)")
            else:
                logger.warning(f"  Found {local_id} → {avi_id} but state={state}")

    if args.auto:
        dirs_to_process = find_output_dirs_missing_jsons()
        video_ids = [Path(d).name for d in dirs_to_process]
        logger.info(f"Auto-discovered {len(video_ids)} directories missing JSONs: {video_ids}")
    elif args.video_ids:
        video_ids = args.video_ids
    else:
        parser.print_help()
        return

    results = {"success": [], "failed": [], "not_found": []}

    for vid in video_ids:
        output_dir = f"outputs/{vid}"
        if vid not in avi_name_map:
            logger.error(f"  ✗ {vid}: No matching processed video found in AVI account")
            results["not_found"].append(vid)
            continue

        avi_video_id = avi_name_map[vid]
        try:
            save_avi_results(client, avi_video_id, output_dir, vid)
            results["success"].append(vid)
        except Exception as e:
            logger.error(f"  ✗ {vid}: Failed to fetch — {e}")
            results["failed"].append(vid)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Success:   {len(results['success'])}  {results['success']}")
    logger.info(f"  Failed:    {len(results['failed'])}  {results['failed']}")
    logger.info(f"  Not found: {len(results['not_found'])}  {results['not_found']}")


if __name__ == "__main__":
    main()
