from pathlib import Path

from src.Preprocessor import Preprocessor
from src.BagofWords import BagOfWords


def main():

    print("====" * 10)
    print("No Preprocessing + COUNTS")
    print("====" * 10)
    pre = Preprocessor(remove_stopwords=False, remove_names=False, remove_contractions=False)
    docs, y = pre.load_corpus(Path("chapters"))

    bow = BagOfWords(labels=("19th", "20th"))
    bow.fit(docs, y)

    for label in ("19th", "20th"):
        counts = bow.get_counts(label)
        print(f"\nTop 10 raw-count words for {label}:")
        for w, c in counts.most_common(10):
            print(w, c)

    print("====" * 10)
    print("Preprocessing + COUNTS")
    print("====" * 10)
    pre = Preprocessor(remove_stopwords=True, remove_names=True, remove_contractions=True)
    docs, y = pre.load_corpus(Path("chapters"))

    bow = BagOfWords(labels=("19th", "20th"))
    bow.fit(docs, y)

    for label in ("19th", "20th"):
        counts = bow.get_counts(label)
        print(f"\nTop 10 raw-count words for {label}:")
        for w, c in counts.most_common(10):
            print(w, c)

    print("====" * 10)
    print("No Preprocessing + TF-IDF")
    print("====" * 10)
    pre = Preprocessor(remove_stopwords=False, remove_names=False, remove_contractions=False)
    docs, y = pre.load_corpus(Path("chapters"))

    bow = BagOfWords(labels=("19th", "20th"))
    bow.fit_tfidf(docs, y)

    for label in ("19th", "20th"):
        tfidf = bow.get_tfidf(label)
        print(f"\nTop 10 TF-IDF words for {label}:")
        top = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
        for w, c in top:
            print(w, round(c, 4))

    print("====" * 10)  
    print("Preprocessing + TF-IDF")
    print("====" * 10)
    pre = Preprocessor(remove_stopwords=True, remove_names=True, remove_contractions=True)
    docs, y = pre.load_corpus(Path("chapters"))

    bow = BagOfWords(labels=("19th", "20th"))
    bow.fit_tfidf(docs, y)

    for label in ("19th", "20th"):
        tfidf = bow.get_tfidf(label)
        print(f"\nTop 10 TF-IDF words for {label}:")
        top = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
        for w, c in top:
            print(w, round(c, 4))

if __name__ == "__main__":
    main()
