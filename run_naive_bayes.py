from pathlib import Path

from src.Preprocessor import Preprocessor
from src.BagofWords import BagOfWords
from src.NaiveBayes import NaiveBayesWordAnalyzer


def main():
    pre = Preprocessor(remove_stopwords=True, remove_names=True, remove_contractions=True)
    docs, y = pre.load_corpus(Path("chapters"))

    bow = BagOfWords(labels=("19th", "20th"))
    bow.fit_tfidf(docs, y)

    tfidf_19 = bow.get_tfidf("19th")
    tfidf_20 = bow.get_tfidf("20th")

    nb = NaiveBayesWordAnalyzer(tfidf_19, tfidf_20)
    s19, s20 = nb.llr_scores()

    print("\nTop 10 words for 19th century:")
    for w, s in nb.top_k(s19, k=10):
        print(w, round(s, 3))

    print("\nTop 10 words for 20th century:")
    for w, s in nb.top_k(s20, k=10):
        print(w, round(s, 3))


if __name__ == "__main__":
    main()
