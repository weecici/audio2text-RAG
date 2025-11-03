from typing import Any
from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    status: int = Field(..., description="HTTP status code of the ingestion process")
    message: str = Field(..., description="Detailed message about the ingestion")


class RetrievalResponse(BaseModel):
    status: int = Field(..., description="HTTP status code of the retrieval process")
    results: list[list[dict[str, Any]]] = Field(
        ..., description="List of retrieved documents with their metadata"
    )


class GenerationResponse(BaseModel):
    status: int = Field(..., description="HTTP status code of the generation process")
    responses: list[str] = Field(
        ..., description="List of generated responses corresponding to the queries"
    )
