from functools import lru_cache
from qdrant_client import QdrantClient, models
from llama_index.core.schema import BaseNode
from src.core import config


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=30)


def ensure_collection_exists(
    collection_name: str,
    dense_name: str = config.EMBEDDING_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    vector_size: int = config.EMBEDDING_DIM,
) -> None:
    client = get_qdrant_client()

    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_name: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=100,
                ),
            ),
        },
        sparse_vectors_config={
            sparse_name: models.SparseVectorParams(),
        },
    )
    print(f"Created collection '{collection_name}'")


def upsert_nodes(
    nodes: list[BaseNode],
    collection_name: str,
    dense_name: str = config.EMBEDDING_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
    vector_size: int = config.EMBEDDING_DIM,
) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upserting")

    if not all(node.embedding is not None for node in nodes):
        raise ValueError("All nodes must have embeddings attached before upserting")

    ensure_collection_exists(
        collection_name=collection_name,
        dense_name=dense_name,
        sparse_name=sparse_name,
        vector_size=vector_size,
    )

    client = get_qdrant_client()
    out = client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=node.id_,
                vector={
                    dense_name: node.embedding,
                },
                payload={
                    "text": node.text,
                    "metadata": node.metadata,
                },
            )
            for node in nodes
        ],
    )

    # Qdrant returns UpdateStatus.ACKNOWLEDGED or COMPLETED
    if out.status not in (
        models.UpdateStatus.ACKNOWLEDGED,
        models.UpdateStatus.COMPLETED,
    ):
        raise RuntimeError(f"Failed to upsert nodes: {out}")
