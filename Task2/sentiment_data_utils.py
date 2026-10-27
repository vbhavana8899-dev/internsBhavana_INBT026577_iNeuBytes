"""
sentiment_data_utils.py
Shared data loading/cleaning for Task 2 Part A and Part B.
Dataset: NLTK's movie_reviews corpus (2000 IMDb-sourced reviews, balanced).
Both parts import from here so they use the IDENTICAL seed and train/test split.
"""

import re
import string
import random

import nltk
from nltk.corpus import movie_reviews, stopwords

SEED = 42


def ensure_nltk_data():
    for pkg, path in [
        ("movie_reviews", "corpora/movie_reviews"),
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean_text(text: str, stop_words: set) -> str:
    """Lowercase, strip punctuation, remove stopwords, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and t.strip()]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def load_sentiment_data(test_size: float = 0.2):
    """
    Returns (x_train_raw, y_train), (x_test_raw, y_test) as cleaned strings
    and 0/1 labels (0=neg, 1=pos), using a fixed seed/split so Part A and
    Part B are directly comparable.
    """
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    docs = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    random.seed(SEED)
    random.shuffle(docs)

    texts = [clean_text(text, stop_words) for text, _ in docs]
    labels = [1 if cat == "pos" else 0 for _, cat in docs]

    split_idx = int(len(texts) * (1 - test_size))
    x_train, x_test = texts[:split_idx], texts[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    return (x_train, y_train), (x_test, y_test)


def class_distribution(labels):
    pos = sum(labels)
    neg = len(labels) - pos
    return {"positive": pos, "negative": neg, "total": len(labels)}
