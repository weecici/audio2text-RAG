import os
import bm25s
import nltk
import numpy as np
from scipy.sparse import csc_matrix
from nltk import WordNetLemmatizer, SnowballStemmer

nltk.download("wordnet")

# Create your corpus here
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]


def lemmatize(words: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def stem(words: list[str]) -> list[str]:
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in words]


method = lemmatize

# Tokenize the corpus and only keep the ids (faster and saves memory)
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=method)
vocab = corpus_tokens.vocab.copy()

# Create the BM25 model and index the corpus
retriever = bm25s.BM25(method="robertson", k1=1.5, b=0.75)
retriever.index(corpus_tokens)


# Query the corpus
query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query, stemmer=method)

# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
# To return docs instead of IDs, set the `corpus=corpus` parameter.
results, scores = retriever.retrieve(query_tokens, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")
