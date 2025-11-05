import psycopg
from typing import Optional, Literal
from psycopg import sql
from pgvector import Vector, SparseVector
from pgvector.psycopg import register_vector
from src import schemas
from src.core import config
from src.services.internal import fuse_results
from src.utils import *
from .storage import (
    get_pg_conn,
    ensure_collection_exists,
    POSTINGS_LIST_TABLE_SUFFIX,
    DOC_FREQ_TABLE_SUFFIX,
)


def _rows_to_results(
    rows: list[tuple],
    *,
    distance_to_similarity: Optional[callable] = None,
) -> list[schemas.RetrievedDocument]:
    results: list[schemas.RetrievedDocument] = []
    for row in rows:
        # row: (id, score, text, document_id, title, file_name, file_path)
        (rid, score, text, document_id, title, file_name, file_path) = row
        # Convert distance to similarity if requested
        sim_score = float(score)
        if distance_to_similarity is not None:
            try:
                sim_score = float(distance_to_similarity(sim_score))
            except Exception:
                sim_score = float(score)
        payload = schemas.DocumentPayload(
            text=text,
            metadata=schemas.DocumentMetadata(
                document_id=document_id or "",
                title=title or "",
                file_name=file_name or "",
                file_path=file_path or "",
            ),
        )
        results.append(
            schemas.RetrievedDocument(
                id=str(rid),
                score=sim_score,
                payload=payload,
            )
        )
    return results


def dense_search(
    query_embeddings: list[list[float]],
    collection_name: str,
    top_k: int = 5,
    dense_name: str = config.DENSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    conn = get_pg_conn()
    ensure_collection_exists(collection_name=collection_name, dense_name=dense_name)

    logger.info(
        f"pgvector.dense_search collection={collection_name} top_k={top_k} using={dense_name} op=<=> (cosine distance)"
    )

    query_tmpl = sql.SQL(
        """
		SELECT id,
			   {dense_col} <=> %s AS score,
			   text,
			   document_id,
			   title,
			   file_name,
			   file_path
		FROM {table}
		ORDER BY {dense_col} <=> %s
		LIMIT %s;
		"""
    ).format(
        table=sql.Identifier(collection_name),
        dense_col=sql.Identifier(dense_name),
    )

    all_results: list[list[schemas.RetrievedDocument]] = []
    with conn.cursor() as cur:
        for emb in query_embeddings:
            vec = Vector(emb)
            cur.execute(query_tmpl, (vec, vec, top_k))
            rows = cur.fetchall()
            if not rows:
                logger.debug("pgvector.dense_search: 0 candidates")
            else:
                preview = min(5, len(rows))
                for i in range(preview):
                    rid, dist, *_ = rows[i]
                    try:
                        d = float(dist)
                    except Exception:
                        d = 0.0
                    logger.debug(
                        f"dense cand[{i}] id={rid} dist={d:.6f} sim={1.0 - d:.6f}"
                    )
            # cosine distance -> similarity in [-1, 1] via (1 - distance)
            all_results.append(
                _rows_to_results(rows, distance_to_similarity=lambda d: 1.0 - d)
            )

    return all_results


def sparse_search(
    query_embeddings: list[tuple[list[int], list[float]]],
    collection_name: str,
    top_k: int = 5,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    conn = get_pg_conn()
    ensure_collection_exists(collection_name=collection_name, sparse_name=sparse_name)

    logger.info(
        f"pgvector.sparse_search collection={collection_name} top_k={top_k} using={sparse_name} dim={config.SPARSE_DIM} op=<#> (inner product)"
    )

    query_tmpl = sql.SQL(
        """
		SELECT id,
               {sparse_col} <#> %s AS score,
			   text,
			   document_id,
			   title,
			   file_name,
			   file_path
		FROM {table}
        ORDER BY {sparse_col} <#> %s
		LIMIT %s;
		"""
    ).format(
        table=sql.Identifier(collection_name),
        sparse_col=sql.Identifier(sparse_name),
    )

    all_results: list[list[schemas.RetrievedDocument]] = []
    with conn.cursor() as cur:
        for indices, values in query_embeddings:
            if len(indices) == 0:
                all_results.append([])
                continue
            vec = SparseVector(
                {int(i): float(v) for i, v in zip(indices, values)}, config.SPARSE_DIM
            )
            cur.execute(query_tmpl, (vec, vec, top_k))
            rows = cur.fetchall()

            if not rows:
                logger.debug("pgvector.sparse_search: 0 candidates")
            else:
                preview = min(5, len(rows))
                for i in range(preview):
                    rid, neg_ip, *_ = rows[i]
                    try:
                        nip = float(neg_ip)
                    except Exception:
                        nip = 0.0
                    logger.debug(
                        f"sparse cand[{i}] id={rid} neg_ip={nip:.6f} ip={-nip:.6f}"
                    )

            # convert to positive similarity = -value (sum of weights)
            all_results.append(
                _rows_to_results(rows, distance_to_similarity=lambda v: -v)
            )

    return all_results


def index_search(
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
    method: Literal["tfidf", "okapi-bm25"] = "okapi-bm25",
) -> list[list[schemas.RetrievedDocument]]:
    conn = get_pg_conn()
    ensure_collection_exists(collection_name=collection_name)

    results_all: list[list[schemas.RetrievedDocument]] = []

    # Precompute corpus stats: N and avg_dl
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT COUNT(*)::int, AVG(doc_len) FROM {};").format(
                sql.Identifier(collection_name)
            )
        )
        row = cur.fetchone()
        N = int(row[0]) if row and row[0] is not None else 0
        avg_dl = float(row[1]) if row and row[1] is not None else 0.0

    if N == 0:
        return [[] for _ in query_texts]

    # Process each query independently
    for query_text in query_texts:
        tokens: list[str] = tokenize([query_text], return_ids=False)[0]
        if not tokens:
            results_all.append([])
            continue

        uniq_terms = sorted(set(tokens))

        # Fetch document frequencies for query terms
        df_map: dict[str, int] = {}
        pl_table = f"{collection_name}_{POSTINGS_LIST_TABLE_SUFFIX}"
        df_table = f"{collection_name}_{DOC_FREQ_TABLE_SUFFIX}"

        # Build placeholders for IN clause
        placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(uniq_terms))

        with conn.cursor() as cur:
            # df per term
            df_query = sql.SQL(
                "SELECT term, doc_freq FROM {} WHERE term IN ({});"
            ).format(sql.Identifier(df_table), placeholders)
            cur.execute(df_query, uniq_terms)
            for term, doc_freq in cur.fetchall() or []:
                df_map[term] = int(doc_freq)

        if not df_map:
            results_all.append([])
            continue

        # Compute IDF per term using provided util
        idf_map: dict[str, float] = {t: calc_idf(df_map[t], N) for t in df_map}

        # Fetch postings joined with doc_len for those terms
        doc_scores: dict[object, float] = {}
        doc_lens: dict[object, int] = {}

        with conn.cursor() as cur:
            postings_query = sql.SQL(
                """
                SELECT p.term, p.doc_id, p.freq, m.doc_len
                FROM {pl} p
                JOIN {main} m ON m.id = p.doc_id
                WHERE p.term IN ({terms})
                """
            ).format(
                pl=sql.Identifier(pl_table),
                main=sql.Identifier(collection_name),
                terms=placeholders,
            )
            cur.execute(postings_query, uniq_terms)
            rows = cur.fetchall() or []

        # Aggregate scores per document
        if method == "tfidf":
            for term, doc_id, tf_doc, dl in rows:
                idf = idf_map.get(term)
                if idf is None:
                    continue
                score = calc_tfidf(int(tf_doc), float(idf))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                if doc_id not in doc_lens:
                    doc_lens[doc_id] = int(dl)
        elif method == "okapi-bm25":  # okapi-bm25
            for term, doc_id, tf_doc, dl in rows:
                idf = idf_map.get(term)
                if idf is None:
                    continue
                score = calc_okapi_bm25(
                    tf=int(tf_doc), idf=float(idf), dl=int(dl), avg_dl=float(avg_dl)
                )
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                if doc_id not in doc_lens:
                    doc_lens[doc_id] = int(dl)
        else:
            raise ValueError(f"Unsupported scoring method: {method}")

        if not doc_scores:
            results_all.append([])
            continue

        # Select top_k docs by score
        top_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[
            :top_k
        ]
        top_ids = [doc_id for doc_id, _ in top_docs]

        # Fetch payloads for top docs
        placeholders_ids = sql.SQL(", ").join([sql.Placeholder()] * len(top_ids))
        with conn.cursor() as cur:
            payload_query = sql.SQL(
                """
                SELECT id, text, document_id, title, file_name, file_path
                FROM {main}
                WHERE id IN ({ids})
                """
            ).format(main=sql.Identifier(collection_name), ids=placeholders_ids)
            cur.execute(payload_query, top_ids)
            payload_rows = cur.fetchall() or []

        payload_map = {
            row[0]: (row[1], row[2], row[3], row[4], row[5]) for row in payload_rows
        }

        # Build results in the correct order
        query_results: list[schemas.RetrievedDocument] = []
        for doc_id, score in top_docs:
            text, document_id, title, file_name, file_path = payload_map.get(
                doc_id, ("", "", "", "", "")
            )
            payload = schemas.DocumentPayload(
                text=text,
                metadata=schemas.DocumentMetadata(
                    document_id=document_id or "",
                    title=title or "",
                    file_name=file_name or "",
                    file_path=file_path or "",
                ),
            )
            query_results.append(
                schemas.RetrievedDocument(
                    id=str(doc_id), score=float(score), payload=payload
                )
            )

        results_all.append(query_results)

    return results_all


def hybrid_search(
    dense_query_embeddings: list[list[float]],
    sparse_query_embeddings: list[tuple[list[int], list[float]]],
    collection_name: str,
    top_k: int = 5,
    overfetch_mul: float = 2.0,
    fusion_method: Literal["dbsf", "rrf"] = config.FUSION_METHOD,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    # Overfetch separately then fuse client-side
    overfetch_amount = max(top_k, int(top_k * overfetch_mul))

    dense_results = dense_search(
        query_embeddings=dense_query_embeddings,
        collection_name=collection_name,
        top_k=overfetch_amount,
        dense_name=dense_name,
    )
    sparse_results = sparse_search(
        query_embeddings=sparse_query_embeddings,
        collection_name=collection_name,
        top_k=overfetch_amount,
        sparse_name=sparse_name,
    )

    fused_results: list[list[schemas.RetrievedDocument]] = []
    for d_res, s_res in zip(dense_results, sparse_results):
        fused = fuse_results(results1=d_res, results2=s_res, method=fusion_method)
        fused_results.append(fused[:top_k])

    return fused_results


def ii_hybrid_search(
    dense_query_embeddings: list[list[float]],
    query_texts: list[str],
    collection_name: str,
    top_k: int = 5,
    overfetch_mul: float = 2.0,
    fusion_method: Literal["dbsf", "rrf"] = config.FUSION_METHOD,
    dense_name: str = config.DENSE_MODEL,
    sparse_name: str = config.SPARSE_MODEL,
) -> list[list[schemas.RetrievedDocument]]:
    # Overfetch separately then fuse client-side
    overfetch_amount = max(top_k, int(top_k * overfetch_mul))

    dense_results = dense_search(
        query_embeddings=dense_query_embeddings,
        collection_name=collection_name,
        top_k=overfetch_amount,
        dense_name=dense_name,
    )
    sparse_results = index_search(
        query_texts=query_texts,
        collection_name=collection_name,
        top_k=overfetch_amount,
    )

    fused_results: list[list[schemas.RetrievedDocument]] = []
    for d_res, s_res in zip(dense_results, sparse_results):
        fused = fuse_results(results1=d_res, results2=s_res, method=fusion_method)
        fused_results.append(fused[:top_k])

    return fused_results
