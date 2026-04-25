import sys
import logging
from data_loader import VideoDataLoader
from graph_store.connection import MultiGraphManager
from graph_store.builders.clip_builder import ClipGraphBuilder
from graph_store.builders.entity_builder import EntityGraphBuilder
from graph_store.builders.mapping_builder import MappingGraphBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUTS_DIR = "outputs/"
CHUNK_A = 10.0
CHUNK_B = 12.0

def run_graph_pipeline_for_video(outputs_dir: str):
    logger.info(f"Loading data from {outputs_dir}...")
    loader = VideoDataLoader(outputs_dir)
    data = loader.load_data()
    
    if not data:
        logger.error("No data loaded. Please check the outputs directory.")
        return

    logger.info("Initializing Neo4j Graph Connections...")
    with MultiGraphManager() as manager:
        
        # 1. Build Clip Graph
        logger.info("="*50)
        logger.info("Building Layer 1: Temporal Clip Graph (Instance 1)...")
        clip_builder = ClipGraphBuilder(manager.clip_graph)
        clip_builder.build_graph(data, outputs_dir, a=CHUNK_A, b=CHUNK_B)
        logger.info("Layer 1 completed.")
        
        # 2. Build Entity Graph
        logger.info("="*50)
        logger.info("Building Layer 2: Semantic Entity Graph (Instance 2)...")
        entity_builder = EntityGraphBuilder(manager.entity_graph)
        entity_builder.build_graph(data)
        logger.info("Layer 2 completed.")
        
        # 3. Build Mapping Graph
        logger.info("="*50)
        logger.info("Building Layer 3: Cross-Graph Mapping (Instance 3)...")
        mapping_builder = MappingGraphBuilder(manager.mapping_graph)
        mapping_builder.build_graph(data, clip_graph_conn=manager.clip_graph)
        logger.info("Layer 3 completed.")
        
        logger.info("="*50)
        logger.info("Neo4j Graph Construction Pipeline Finished Successfully!")

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else OUTPUTS_DIR
    run_graph_pipeline_for_video(target_dir)

if __name__ == "__main__":
    main()
