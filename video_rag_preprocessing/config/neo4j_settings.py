import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: Optional[str] = None

class GraphSettings:
    """
    Settings for the three separate Neo4j instances.
    Requires environment variables to be set.
    """
    
    @classmethod
    def get_clip_graph_config(cls) -> Neo4jConfig:
        return Neo4jConfig(
            uri=os.getenv("NEO4J_CLIP_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_CLIP_USER", "neo4j"),
            password=os.getenv("NEO4J_CLIP_PASSWORD", "password"),
            database=os.getenv("NEO4J_CLIP_DATABASE", None)
        )

    @classmethod
    def get_entity_graph_config(cls) -> Neo4jConfig:
        return Neo4jConfig(
            uri=os.getenv("NEO4J_ENTITY_URI", "bolt://localhost:7688"),
            username=os.getenv("NEO4J_ENTITY_USER", "neo4j"),
            password=os.getenv("NEO4J_ENTITY_PASSWORD", "password"),
            database=os.getenv("NEO4J_ENTITY_DATABASE", None)
        )

    @classmethod
    def get_mapping_graph_config(cls) -> Neo4jConfig:
        return Neo4jConfig(
            uri=os.getenv("NEO4J_MAPPING_URI", "bolt://localhost:7689"),
            username=os.getenv("NEO4J_MAPPING_USER", "neo4j"),
            password=os.getenv("NEO4J_MAPPING_PASSWORD", "password"),
            database=os.getenv("NEO4J_MAPPING_DATABASE", None)
        )
