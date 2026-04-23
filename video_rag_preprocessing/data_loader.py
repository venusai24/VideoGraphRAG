import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class VideoDataLoader:
    """
    A unified data loader for the two-layer GraphRAG system.
    Parses project metadata and sequentially loads all JSON payloads for each clip.
    """
    def __init__(self, outputs_dir: str | Path):
        self.outputs_dir = Path(outputs_dir)
        self.project_index_path = self.outputs_dir / "projects" / "project_yvryjpdpll_index.json"
        
        # Mapping: video_id -> local_folder_name (e.g., 'clip_0')
        self.video_to_folder: Dict[str, str] = {}
        
        # Structured data: { 'clip_0': { 'keywords': {...}, 'ocr': {...}, ... } }
        self.clip_data: Dict[str, Dict[str, Any]] = {}

    def _parse_project_index(self) -> None:
        """
        Parses project_index.json to create a video_id to local_folder_name mapping.
        Falls back to directory scanning if index file is missing or lacks mappings.
        """
        # If the hardcoded index doesn't exist, try to find any *_index.json in projects/
        if not self.project_index_path.exists():
            projects_dir = self.outputs_dir / "projects"
            if projects_dir.exists():
                indices = list(projects_dir.glob("*_index.json"))
                if indices:
                    self.project_index_path = indices[0]
                    logger.info(f"Using discovered index file: {self.project_index_path}")

        if self.project_index_path.exists():
            with open(self.project_index_path, 'r', encoding='utf-8') as f:
                try:
                    index_data = json.load(f)
                    # Handle variations in potential index schemas:
                    # Assuming it could be a list of dicts, or a dict with a 'videos' list.
                    videos_list = index_data.get('videos', index_data) if isinstance(index_data, dict) else index_data
                    
                    if isinstance(videos_list, list):
                        for item in videos_list:
                            if isinstance(item, dict):
                                vid_id = item.get("video_id") or item.get("id")
                                folder_name = item.get("local_folder_name") or item.get("folder")
                                
                                if vid_id and folder_name:
                                    self.video_to_folder[str(vid_id)] = str(folder_name)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse project index JSON: {e}")
        else:
            logger.info(f"No project index file found. Skipping index parsing.")
            
        if not self.video_to_folder:
            logger.info("No mappings found in index file. Attempting auto-discovery of clip folders...")
            self._auto_discover_clips()

    def _auto_discover_clips(self) -> None:
        """
        Scans the outputs directory for 'clip_*' folders and extracts video IDs from their contents.
        """
        if not self.outputs_dir.exists():
            logger.error(f"Outputs directory not found: {self.outputs_dir}")
            return

        for item in self.outputs_dir.iterdir():
            # Only process directories starting with 'clip_'
            if item.is_dir() and item.name.startswith("clip_"):
                video_id = None
                
                # Check for video_id in raw_insights.json
                insights_path = item / "raw_insights.json"
                if insights_path.exists():
                    try:
                        with open(insights_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            video_id = data.get("id") or data.get("video_id")
                    except Exception:
                        pass
                
                # Fallback to rag_chunks.json
                if not video_id:
                    chunks_path = item / "rag_chunks.json"
                    if chunks_path.exists():
                        try:
                            with open(chunks_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list) and len(data) > 0:
                                    video_id = data[0].get("video_id") or data[0].get("id")
                        except Exception:
                            pass
                
                # Use folder name as final fallback for ID
                final_id = str(video_id) if video_id else item.name
                self.video_to_folder[final_id] = item.name
                logger.info(f"Mapped {item.name} to video_id: {final_id}")

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Sequentially loads all the JSON files for each clip into memory.
        
        Returns:
            A structured dictionary where keys are the folder names and 
            values are the parsed JSON payloads.
        """
        self._parse_project_index()
        
        expected_files = [
            "keywords.json",
            "ocr.json",
            "rag_chunks.json",
            "raw_insights.json",
            "scenes.json",
            "transcript.json"
        ]
        
        for video_id, folder_name in self.video_to_folder.items():
            clip_dir = self.outputs_dir / folder_name
            
            if not clip_dir.exists() or not clip_dir.is_dir():
                logger.warning(f"Clip directory does not exist: {clip_dir}")
                continue
                
            self.clip_data[folder_name] = {}
            
            for file_name in expected_files:
                file_path = clip_dir / file_name
                payload_key = file_path.stem  # e.g., 'keywords' from 'keywords.json'
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.clip_data[folder_name][payload_key] = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON in {file_path}: {e}")
                        self.clip_data[folder_name][payload_key] = None
                else:
                    # File might be missing for some clips
                    self.clip_data[folder_name][payload_key] = None
                    
        return self.clip_data

if __name__ == "__main__":
    # Example usage:
    # Set the outputs_dir to the actual path of the outputs directory
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Defaults to 'outputs' in the current working directory, 
    # or you can pass the path as a command-line argument.
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    
    try:
        loader = VideoDataLoader(target_dir)
        data = loader.load_data()
        
        print(f"Successfully loaded data for {len(data)} clips.")
        for folder, payloads in data.items():
            loaded_files = [k for k, v in payloads.items() if v is not None]
            print(f"  - {folder}: loaded {len(loaded_files)} files ({', '.join(loaded_files)})")
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
