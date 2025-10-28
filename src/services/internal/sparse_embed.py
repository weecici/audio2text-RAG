import os
import bm25s
import numpy as np
import nltk
from nltk import WordNetLemmatizer, SnowballStemmer
from scipy.sparse import csc_matrix, csr_matrix
from typing import Literal, Optional
from src.utils import tokenize
from src.core import config

nltk.download("wordnet")


def sparse_encode(
    texts: list[str],
    word_process_method: Literal["lemmatize", "stem"] = config.WORD_PROCESS_METHOD,
    bm25_method: str = "robertson",
    bm25_idf_method: Optional[str] = None,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 1.5,
) -> tuple[csr_matrix, dict[str, int]]:

    text_tokens = tokenize(texts=texts, word_process_method=word_process_method)
    vocab = text_tokens.vocab.copy()

    retriever = bm25s.BM25(
        method=bm25_method, idf_method=bm25_idf_method, k1=k1, b=b, delta=delta
    )
    retriever.index(text_tokens)

    TEMP_STORAGE_PATH = "./.storage/tmp_bm25_index"

    retriever.save(TEMP_STORAGE_PATH)
    data = np.load(os.path.join(TEMP_STORAGE_PATH, "data.csc.index.npy"), mmap_mode="r")
    indices = np.load(
        os.path.join(TEMP_STORAGE_PATH, "indices.csc.index.npy"), mmap_mode="r"
    )
    indptr = np.load(
        os.path.join(TEMP_STORAGE_PATH, "indptr.csc.index.npy"), mmap_mode="r"
    )

    sparse_embeddings = csc_matrix(
        (data, indices, indptr), shape=(len(texts), len(vocab))
    ).tocsr()

    return sparse_embeddings, vocab
