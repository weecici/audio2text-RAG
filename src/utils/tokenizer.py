import nltk
import bm25s
from bm25s.tokenization import Tokenized
from nltk import WordNetLemmatizer, SnowballStemmer
from typing import Literal

nltk.download("wordnet")


def stem(words: list[str]) -> list[str]:
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in words]


def lemmatize(words: list[str]) -> list[str]:
    lmtz = WordNetLemmatizer()
    return [lmtz.lemmatize(word) for word in words]


def tokenize(
    texts: list[str],
    word_process_method: Literal["lemmatize", "stem"] = "stem",
    return_ids: bool = True,
) -> list[list[str]] | Tokenized:
    process_method: callable[[list[str]], list[str]] = stem
    if word_process_method == "lemmatize":
        process_method = lemmatize
    text_tokens = bm25s.tokenize(
        texts=texts, stopwords="en", stemmer=process_method, return_ids=return_ids
    )
    return text_tokens
