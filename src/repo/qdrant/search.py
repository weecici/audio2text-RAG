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
            with_vector=True,
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
            node_content_str = payload.get("_node_content")
            if node_content_str:
                node_content: dict = json.loads(node_content_str)

                important_content = {
                    k: node_content.get(k) for k in ["id_", "metadata", "text"]
                }
                important_content["score"] = scored_point.score

                current_result.append(important_content)
            else:
                raise ValueError(f"Missing node's contents")

        all_results.append(current_result)

    return all_results
