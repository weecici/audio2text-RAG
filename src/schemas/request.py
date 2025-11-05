from pydantic import BaseModel, Field
from typing import Literal


class IngestionRequest(BaseModel):
    file_paths: list[str] = Field(
        default=[], description="List of file paths to ingest documents from"
    )
    file_dir: str = Field(
        default="", description="Directory containing files to ingest"
    )
    collection_name: str = Field(
        default="documents", description="Name of the Qdrant collection"
    )


class RetrievalRequest(BaseModel):
    queries: list[str] = Field(..., description="The list of query texts")
    collection_name: str = Field(
        default="documents", description="Name of the Qdrant collection"
    )
    top_k: int = Field(
        default=5, description="Number of top similar documents to retrieve"
    )
    mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid", description="The retrieval mode to use"
    )
    sparse_search_method: Literal["embedding", "inverted_index"] = Field(
        default="embedding",
        description="Method for sparse search",
    )
    overfetch_mul: float = Field(
        default=2.0,
        description="Multiplier for overfetching in hybrid retrieval",
    )
    rerank_enabled: bool = Field(
        default=False, description="Whether to rerank retrieved results"
    )


class GenerationRequest(RetrievalRequest):
    model_name: Literal[
        "gpt-oss-120b",
        "llama-3.3-70b",
        "qwen-3-235b-a22b-thinking-2507",
        "qwen-3-coder-480b",
    ] = Field(
        default="gpt-oss-120b",
        description="Name of the LLM to use for generation",
    )
