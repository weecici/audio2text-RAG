import json
from llama_index.core.schema import BaseNode, TextNode
from qdrant_client import models
from .storage import get_qdrant_client


def dense_search(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    filters: models.Filter = None,
) -> list[tuple[list[BaseNode], list[float]]]:

    client = get_qdrant_client()

    search_queries = [
        models.SearchRequest(
            vector=query_emb,
            limit=top_k,
            filter=filters,
            with_payload=True,
            with_vector=False,
        )
        for query_emb in query_embeddings
    ]

    batch_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries,
    )

    all_results = []
    for search_result in batch_results:
        current_result = []

        for scored_point in search_result:
            payload = scored_point.payload

            important_content = {
                "id": scored_point.id,
                "score": scored_point.score,
                **payload,
            }

            current_result.append(important_content)

        all_results.append(current_result)

    return all_results
