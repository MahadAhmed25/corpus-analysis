from collections import Counter
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

class BagOfWords:
    """
    Baseline BoW:
    - class-level word counts (19th vs 20th)
    """

    def __init__(self, labels=("19th", "20th")):
        self.labels = tuple(labels)
        self.class_counts: Dict[str, Counter] = {l: Counter() for l in self.labels}

    def fit(self, docs, y):

        for tokens, label in zip(docs, y):
            self.class_counts[label].update(tokens)

    def get_counts(self, label):
        return self.class_counts[label]

    def fit_tfidf(self, docs, y):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            lowercase=False,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        self.tfidf_labels = y

    def get_tfidf(self, label):
        idxs = [i for i, l in enumerate(self.tfidf_labels) if l == label]
        sub = self.tfidf_matrix[idxs]

        avg = sub.mean(axis=0)
        scores = avg.A1

        vocab = self.vectorizer.get_feature_names_out()
        return dict(zip(vocab, scores))
