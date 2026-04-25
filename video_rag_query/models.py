from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class Entity(BaseModel):
    name: str = Field(..., description="The extracted name of the entity.")
    type: str = Field(..., description="The type/class of the entity.")
    resolved_entity_id: Optional[str] = Field(None, description="Placeholder for the canonical EntityRef ID to be populated post-extraction.")

class TemporalConstraints(BaseModel):
    relation: Literal["before", "after", "during", "none"] = Field("none", description="The temporal relationship.")
    anchor_event: Optional[str] = Field(None, description="The anchor event string if applicable.")
    direction: Literal["forward", "backward", "neutral", "none"] = Field("none", description="The traversal direction.")

class SubQuery(BaseModel):
    id: str = Field(..., description="A unique identifier for this sub-query, e.g., 'Q1'.")
    type: str = Field(..., description="The type of sub-query, e.g., 'temporal_traversal', 'event_localization'.")
    goal: str = Field(..., description="A clear description of what this sub-query aims to achieve.")
    required_graph_components: List[Literal["APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"]] = Field(
        ..., description="The graph components required to execute this sub-query."
    )

class QueryDecomposition(BaseModel):
    query_type: str = Field(..., description="The classified type of the query.")
    entities: List[Entity] = Field(..., description="List of extracted entities.")
    actions: List[str] = Field(..., description="List of extracted actions or relations.")
    temporal_constraints: TemporalConstraints = Field(..., description="Temporal constraints for the query.")
    sub_queries: List[SubQuery] = Field(..., description="Ordered list of sub-queries for execution.")
    execution_plan: List[str] = Field(..., description="Ordered traversal steps referencing explicit graph edges.")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0.")
    ambiguity_flags: List[str] = Field(default_factory=list, description="List of any ambiguities identified.")

class FailureResponse(BaseModel):
    status: str = Field("failure", description="Status of the operation.")
    reason: str = Field(..., description="Reason for failure.")
    fallback: Optional[Any] = Field(None, description="Fallback information if any.")
