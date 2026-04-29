import logging
import os
from data_loader import VideoDataLoader
from graph_store.connection import MultiGraphManager
from graph_store.builders.entity_builder import EntityGraphBuilder
from graph_store.builders.mapping_builder import MappingBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ingestion(video_id):
    outputs_dir = "/mnt/MIG_store/Datasets/blending/madhav/VRAG/outputs"
    target_video_dir = os.path.join(outputs_dir, video_id)
    
    # 1. Load Data
    logger.info(f"Loading data for {video_id}...")
    loader = VideoDataLoader(outputs_dir)
    # We only want to load data for THIS video
    all_data = loader.load_data()
    video_data = {video_id: all_data.get(video_id)}
    if not video_data[video_id]:
        logger.error(f"No data found for {video_id}")
        return

    with MultiGraphManager() as manager:
        # 2. Build Entity Graph (Graph 2)
        # This will populate RELATED_TO and SUBCLASS_OF edges
        logger.info(f"Populating Entity Graph (Graph 2) for {video_id}...")
        entity_builder = EntityGraphBuilder(manager.entity_graph)
        entity_builder.build_graph(video_data)
        
        # 3. Build Mapping (SQLite)
        # First, fetch clip intervals from Graph 1
        logger.info(f"Fetching clip intervals from Clip Graph (Graph 1) for {video_id}...")
        query = "MATCH (c:Clip {video_id: $vid}) RETURN c.id as node_id, c.video_id as video_id, c.start as start, c.end as end"
        clip_intervals = manager.clip_graph.execute_query(query, {"vid": video_id})
        
        if not clip_intervals:
            logger.warning(f"No clips found in Graph 1 for {video_id}. Mapping will be incomplete.")
        
        db_path = os.path.join(target_video_dir, "mapping.db")
        logger.info(f"Populating SQLite Mapping at {db_path}...")
        mapping_builder = MappingBuilder(db_path)
        mapping_builder.build(video_data, clip_intervals, clear_existing=True)

    logger.info("Test ingestion complete.")

if __name__ == "__main__":
    test_ingestion("ac1682fb")
