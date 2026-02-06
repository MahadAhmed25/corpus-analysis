import math
from collections import Counter
from typing import Dict, Tuple


class NaiveBayesWordAnalyzer:
    """
    Computes word probabilities and log-likelihood ratio (LLR)
    between two classes (e.g., 19th vs 20th), using class-level counts.
    """

    def __init__(self, counts_a: Counter, counts_b: Counter):
        self.counts_a = counts_a
        self.counts_b = counts_b

        self.vocab = set(counts_a.keys()) | set(counts_b.keys())
        self.V = len(self.vocab)

        self.total_a = sum(counts_a.values())
        self.total_b = sum(counts_b.values())

    def llr_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        scores_a: Dict[str, float] = {}
        scores_b: Dict[str, float] = {}

        for w in self.vocab:
            p_a = (self.counts_a[w] + 1) / (self.total_a + self.V)
            p_b = (self.counts_b[w] + 1) / (self.total_b + self.V)

            scores_a[w] = math.log(p_a) - math.log(p_b)
            scores_b[w] = math.log(p_b) - math.log(p_a)

        return scores_a, scores_b

    @staticmethod
    def top_k(scores: Dict[str, float], k: int = 10):
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
