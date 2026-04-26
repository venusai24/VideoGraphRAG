import os
import sys
import logging
from data_loader import VideoDataLoader
from graph_store.connection import MultiGraphManager
from graph_store.builders.clip_builder import ClipGraphBuilder
from graph_store.builders.entity_builder import EntityGraphBuilder
from graph_store.builders.mapping_builder import MappingBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUTS_DIR    = "outputs/"
CHUNK_A        = 10.0
CHUNK_B        = 12.0
MAPPING_DB     = "mapping.db"   # written inside outputs_dir

def run_graph_pipeline_for_video(outputs_dir: str):
    logger.info(f"Loading data from {outputs_dir}...")
    loader = VideoDataLoader(outputs_dir)
    data = loader.load_data()
    
    if not data:
        logger.error("No data loaded. Please check the outputs directory.")
        return

    db_path = os.path.join(outputs_dir, MAPPING_DB)

    logger.info("Initializing Neo4j Graph Connections (Clip + Entity only)...")
    with MultiGraphManager() as manager:
        
        # 1. Build Clip Graph
        logger.info("=" * 50)
        logger.info("Building Layer 1: Temporal Clip Graph...")
        clip_builder = ClipGraphBuilder(manager.clip_graph)
        clip_builder.build_graph(data, outputs_dir, a=CHUNK_A, b=CHUNK_B)
        logger.info("Layer 1 completed.")
        
        # 2. Build Entity Graph
        logger.info("=" * 50)
        logger.info("Building Layer 2: Semantic Entity Graph...")
        entity_builder = EntityGraphBuilder(manager.entity_graph)
        entity_builder.build_graph(data)
        logger.info("Layer 2 completed.")
        
        # 3. Fetch clip intervals from Neo4j for overlap calculation
        logger.info("=" * 50)
        logger.info("Fetching clip intervals for bipartite mapping...")
        try:
            records = manager.clip_graph.execute_query(
                "MATCH (c:Clip) RETURN c.id AS id, c.video_id AS video_id, c.start AS start, c.end AS end"
            )
            clip_intervals = [
                {"node_id": r["id"], "video_id": r["video_id"],
                 "start": r["start"], "end": r["end"]}
                for r in records
            ]
            logger.info(f"Fetched {len(clip_intervals)} clip intervals.")
        except Exception as e:
            logger.error(f"Failed to fetch clip intervals: {e}")
            clip_intervals = []

    # 4. Build Mapping (SQLite — outside Neo4j)
    logger.info("=" * 50)
    logger.info(f"Building Layer 3: Entity↔Clip Mapping (SQLite → {db_path})...")
    mapping_builder = MappingBuilder(db_path)
    store = mapping_builder.build(
        data,
        clip_intervals,
        clear_existing=True,
    )
    stats = store.stats()
    store.close()
    logger.info(
        f"Layer 3 completed. entity→clip mappings: {stats['entity_clip_mappings']}, "
        f"clip similarities: {stats['clip_similarities']}. Stored in {db_path}"
    )

    logger.info("=" * 50)
    logger.info("Neo4j Graph Construction Pipeline Finished Successfully!")

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else OUTPUTS_DIR
    run_graph_pipeline_for_video(target_dir)

if __name__ == "__main__":
    main()
