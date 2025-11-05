from collections import Counter
from src import schemas
from src.core import config
from src.utils import tokenize


def build_inverted_index(
    texts: list[str],
    doc_ids: list[str],
    word_process_method: str = config.WORD_PROCESS_METHOD,
) -> tuple[dict[str, schemas.TermEntry], dict[str, int]]:
    """Returns postings list and document lengths for the given texts."""

    postings_list: dict[str, schemas.TermEntry] = {}
    doc_lens: dict[str, int] = {}

    tokenized_docs = tokenize(
        texts=texts, word_process_method=word_process_method, return_ids=False
    )

    # creating postings list
    for doc_id, tokens in zip(doc_ids, tokenized_docs):
        counts = Counter(tokens)
        for token, term_freq in counts.items():
            doc_lens[doc_id] = doc_lens.get(doc_id, 0) + term_freq
            if token not in postings_list:
                postings_list[token] = schemas.TermEntry(doc_freq=0, postings=[])

            postings_list[token].postings.append(
                schemas.PostingEntry(doc_id=doc_id, term_freq=term_freq)
            )

    # compute document frequencies
    for term, term_entry in postings_list.items():
        postings_list[term].doc_freq = len(term_entry.postings)

    return postings_list, doc_lens
