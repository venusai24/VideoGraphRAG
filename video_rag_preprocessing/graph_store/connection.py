import logging
from typing import Dict, Optional, Any
from neo4j import GraphDatabase, Driver
from config.neo4j_settings import GraphSettings, Neo4jConfig

logger = logging.getLogger(__name__)

class GraphConnection:
    """Manages a single Neo4j connection."""
    def __init__(self, config: Neo4jConfig, name: str):
        self.config = config
        self.name = name
        self.driver: Optional[Driver] = None

    def connect(self):
        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.username, self.config.password)
                )
                # Verify connectivity
                self.driver.verify_connectivity()
                logger.info(f"Successfully connected to {self.name} Graph at {self.config.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to {self.name} Graph at {self.config.uri}: {e}")
                raise

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info(f"Closed connection to {self.name} Graph.")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        if not self.driver:
            self.connect()
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
            
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        if not self.driver:
            self.connect()
            
        def _write_transaction(tx):
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]
            
        with self.driver.session(database=self.config.database) as session:
            return session.execute_write(_write_transaction)


class MultiGraphManager:
    """
    Manages connections to the three separate Neo4j graph instances:
    1. Clip Graph
    2. Entity Graph
    3. Mapping Graph
    """
    def __init__(self):
        self.clip_graph = GraphConnection(GraphSettings.get_clip_graph_config(), "Clip")
        self.entity_graph = GraphConnection(GraphSettings.get_entity_graph_config(), "Entity")
        self.mapping_graph = GraphConnection(GraphSettings.get_mapping_graph_config(), "Mapping")

    def connect_all(self):
        """Initializes connections to all three graphs."""
        self.clip_graph.connect()
        self.entity_graph.connect()
        self.mapping_graph.connect()

    def close_all(self):
        """Closes all active connections."""
        self.clip_graph.close()
        self.entity_graph.close()
        self.mapping_graph.close()

    def __enter__(self):
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
