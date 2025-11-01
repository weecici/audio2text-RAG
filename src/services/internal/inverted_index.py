from collections import Counter
from src.core import config
from src.utils import tokenize


def build_inverted_index(
    texts: list[str],
    uuids: list[str],
    metadata: list[dict],
    word_process_method: str = config.WORD_PROCESS_METHOD,
) -> None:
    tokenized = tokenize(
        texts=texts, word_process_method=word_process_method, return_ids=False
    )

    vocab: dict[str, int] = {}
    postings: dict[int, list[list[int]]] = {}
    doc_freqs: dict[int, int] = {}
    meta: dict[str, any] = {}

    for doc_id, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for token, tf in counts.items():
            if token not in vocab:
                vocab[token] = len(vocab)
            idx = vocab[token]
            if idx not in postings:
                postings[idx] = []
            postings[idx].append([doc_id, int(tf)])

    # compute doc_freqs
    for idx, plist in postings.items():
        doc_freqs[idx] = len(plist)

    # convert the key to str (which are actually ints) for json serialization
    postings_serializable = {str(k): v for k, v in postings.items()}

    meta = {
        "doc_count": len(texts),
        "doc_freqs": {str(k): v for k, v in doc_freqs.items()},
        "doc_lens": [len(toks) for toks in tokenized],
        "avg_doc_len": (
            sum(len(toks) for toks in tokenized) / len(tokenized)
            if len(tokenized) > 0
            else None
        ),
        "uuids": uuids,
    }

    indexed_docs: dict[str, dict] = {
        "vocab": vocab,
        "postings": postings_serializable,
        "docs": [
            {"text": text, "metadata": meta} for text, meta in zip(texts, metadata)
        ],
        "meta": meta,
    }

    return indexed_docs
