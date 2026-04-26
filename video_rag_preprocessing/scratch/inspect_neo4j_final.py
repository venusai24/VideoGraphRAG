import logging
from graph_store.connection import MultiGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_graphs():
    with MultiGraphManager() as manager:
        # Inspect Clip Graph
        logger.info("--- Inspecting Clip Graph ---")
        try:
            nodes = manager.clip_graph.execute_query("MATCH (n) RETURN labels(n) as labels, count(*) as count")
            rels = manager.clip_graph.execute_query("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
            logger.info(f"Nodes: {nodes}")
            logger.info(f"Relationships: {rels}")
        except Exception as e:
            logger.error(f"Failed to inspect Clip Graph: {e}")

        # Inspect Entity Graph
        logger.info("--- Inspecting Entity Graph ---")
        try:
            nodes = manager.entity_graph.execute_query("MATCH (n) RETURN labels(n) as labels, count(*) as count")
            rels = manager.entity_graph.execute_query("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
            logger.info(f"Nodes: {nodes}")
            logger.info(f"Relationships: {rels}")
        except Exception as e:
            logger.error(f"Failed to inspect Entity Graph: {e}")

if __name__ == "__main__":
    inspect_graphs()
