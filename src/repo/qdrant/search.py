from typing import Optional
from qdrant_client import models
from .storage import get_qdrant_client
from src.core import config


def dense_search(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    filters: Optional[models.Filter] = None,
    *,
    dense_name: str = config.EMBEDDING_MODEL,
) -> list[list[dict]]:

    client = get_qdrant_client()

    search_queries = []
    for query_emb in query_embeddings:
        search_queries.append(
            models.SearchRequest(
                vector=models.NamedVector(name=dense_name, vector=query_emb),
                limit=top_k,
                filter=filters,
                with_payload=True,
                with_vector=False,
            )
        )

    batch_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries,
    )

    all_results: list[list[dict]] = []
    for search_result in batch_results:
        current_result: list[dict] = []

        for scored_point in search_result:
            payload = scored_point.payload or {}
            current_result.append(
                {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": payload,
                }
            )

        all_results.append(current_result)

    return all_results
