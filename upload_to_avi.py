
import os
import sys
import logging
from dotenv import load_dotenv

# Resolve imports
# Add the current directory and the video_rag_preprocessing directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PREPROC_DIR = os.path.join(SCRIPT_DIR, "video_rag_preprocessing")
sys.path.insert(0, PREPROC_DIR)

from avi_client import AVIClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UploadToAVI")

load_dotenv()

def main():
    # Load credentials from .env
    api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY", "")
    account_id = os.getenv("VIDEO_INDEXER_ACCOUNT_ID", "")
    location = os.getenv("VIDEO_INDEXER_LOCATION", "trial")
    account_type = os.getenv("VIDEO_INDEXER_ACCOUNT_TYPE", "trial")

    if not api_key or not account_id:
        logger.error("Missing AVI credentials in .env file.")
        return

    client = AVIClient(api_key, account_id, location, account_type)

    # Video details
    video_id = "d971a085"
    video_path = os.path.join(SCRIPT_DIR, "input", "The_Debacle_Behind_The_2026_Olympic_Hockey_Rink_[5Elvi_r3ln8]_enc.mp4")
    video_name = f"video_{video_id}"

    if not os.path.exists(video_path):
        logger.error(f"Source video not found: {video_path}")
        return

    logger.info(f"Uploading {video_path} to AVI as '{video_name}'...")
    
    try:
        avi_video_id = client.upload_video(video_path, video_name)
        logger.info(f"Successfully started upload! AVI Video ID: {avi_video_id}")
        logger.info("The video is now being indexed by Azure.")
        logger.info(f"Once processing is complete, you can run: python full_recovery.py {video_id}")
        logger.info("To check status, run: python fetch_avi_results.py --list")
    except Exception as e:
        logger.error(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
