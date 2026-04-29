# Deprecated – not used in final pipeline
import os
import sys
import logging
from pathlib import Path

# Add the video_rag_preprocessing directory to sys.path to import internal modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_rag_preprocessing"))

from graph_store.connection import MultiGraphManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClearNeo4j")

def clear_neo4j_graphs(video_id: str = None):
    """Wipes data from both the Clip Graph and the Entity Graph."""
    logger.info("Connecting to Neo4j instances...")
    try:
        with MultiGraphManager() as manager:
            if video_id:
                logger.info(f"Clearing data for video_id: {video_id}...")
                # Delete Clips for this video and their relationships
                manager.clip_graph.execute_write(
                    "MATCH (n) WHERE n.video_id = $vid OR n.id STARTS WITH $vid DETACH DELETE n",
                    {"vid": video_id}
                )
                # Note: Entity graph nodes are shared across videos, so we usually don't delete them 
                # unless they are only linked to this video. However, the current schema 
                # doesn't strictly track 'source video' on entity nodes themselves easily 
                # since they are normalized. We'll leave entities alone for specific video deletion
                # to avoid breaking other videos' graphs.
                logger.info(f"✓ Data for {video_id} cleared from Clip Graph (Entities preserved).")
            else:
                # 1. Clear Clip Graph
                logger.info("Clearing Clip Graph...")
                manager.clip_graph.execute_write("MATCH (n) DETACH DELETE n")
                logger.info("✓ Clip Graph cleared.")

                # 2. Clear Entity Graph
                logger.info("Clearing Entity Graph...")
                manager.entity_graph.execute_write("MATCH (n) DETACH DELETE n")
                logger.info("✓ Entity Graph cleared.")
            
        logger.info("="*50)
        logger.info("Neo4j cleanup completed.")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"Failed to clear Neo4j graphs: {e}")
        sys.exit(1)

def clear_mapping_files(video_id: str = None):
    """Deletes the entity_clip_mapping.json files."""
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return

    if video_id:
        target = outputs_root / video_id / "entity_clip_mapping.json"
        if target.exists():
            try:
                target.unlink()
                logger.info(f"Deleted: {target}")
            except Exception as e:
                logger.error(f"Failed to delete {target}: {e}")
        else:
            logger.info(f"No mapping file found for video {video_id}")
    else:
        logger.info("Checking for entity_clip_mapping.json files in outputs/...")
        count = 0
        for mapping_file in outputs_root.glob("**/entity_clip_mapping.json"):
            try:
                mapping_file.unlink()
                count += 1
                logger.info(f"Deleted: {mapping_file}")
            except Exception as e:
                logger.error(f"Failed to delete {mapping_file}: {e}")
        
        if count > 0:
            logger.info(f"✓ Deleted {count} mapping files.")
        else:
            logger.info("No mapping files found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wipe existing Neo4j graphs and optionally local mapping files.")
    parser.add_argument("--video-id", type=str, help="Only delete data for this specific video ID (Clip graph only)")
    parser.add_argument("--include-mappings", action="store_true", help="Also delete entity_clip_mapping.json files in outputs/")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()

    if not args.force:
        target_desc = f"video {args.video_id}" if args.video_id else "ALL data"
        confirm = input(f"This will IRREVERSIBLY DELETE {target_desc} in your Neo4j graphs. Are you sure? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    clear_neo4j_graphs(args.video_id)
    
    if args.include_mappings:
        clear_mapping_files(args.video_id)
