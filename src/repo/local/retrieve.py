import os
import json
from collections import Counter
from typing import Literal, Union
from src import schemas
from src.core import config
from src.utils import *
from .storage import _paths_for, _index_cache


def _load_index(collection_name: str) -> dict:
    if collection_name in _index_cache:
        return _index_cache[collection_name]

    paths = _paths_for(collection_name)
    if not os.path.exists(paths["vocab"]) or not os.path.exists(paths["postings"]):
        raise FileNotFoundError(
            f"Index for '{collection_name}' not found on disk at {paths}"
        )

    index: dict[str, dict] = {}
    for key in paths:
        with open(paths[key], "r") as f:
            index[key] = json.load(f)
            if key == "postings":
                # convert keys back to ints
                index[key] = {int(k): v for k, v in index[key].items()}

    _index_cache[collection_name] = index
    return index


def index_retrieve(
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
    method: Literal["tfidf", "okapi-bm25"] = "okapi-bm25",
    word_process_method: str = config.WORD_PROCESS_METHOD,
) -> list[list[schemas.RetrievedDocument]]:

    index = _load_index(collection_name)

    vocab: dict[str, int] = index["vocab"]
    postings: dict[int, list[list[int]]] = index["postings"]
    docs: list[schemas.DocumentPayload] = [
        schemas.DocumentPayload.model_validate(payload) for payload in index["docs"]
    ]
    meta: dict = index["meta"]

    N = meta["doc_count"]

    tokenized_queries = tokenize(
        texts=query_texts, word_process_method=word_process_method, return_ids=False
    )

    results: list[list[schemas.RetrievedDocument]] = []
    for tokens in tokenized_queries:
        counts = Counter(tokens)
        scores: dict[int, float] = {}
        for token, qtf in counts.items():
            if token not in vocab:
                continue
            idx = vocab[token]

            df = meta["doc_freqs"].get(str(idx), 0)
            avg_doc_len = meta["avg_doc_len"]
            idf = calc_idf(df, N)

            for doc_id, tf in postings[idx]:
                scores[doc_id] = scores.get(doc_id, 0.0)
                if method == "tfidf":
                    scores[doc_id] += qtf * calc_tfidf(tf, idf)
                elif method == "okapi-bm25":
                    doc_len = meta["doc_lens"][doc_id]
                    scores[doc_id] += qtf * calc_okapi_bm25(
                        tf, idf, doc_len, avg_doc_len, k1=1.5, b=0.75
                    )
                else:
                    raise ValueError(f"Unknown scoring method: {method}")

        # sort and pick top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        uuids: list[Union[int, str]] = meta["uuids"]

        current_result: list[schemas.RetrievedDocument] = []
        for doc_id, score in sorted_docs:
            current_result.append(
                schemas.RetrievedDocument(
                    id=uuids[doc_id],
                    score=score,
                    payload=docs[doc_id],
                )
            )

        results.append(current_result)

    return results
