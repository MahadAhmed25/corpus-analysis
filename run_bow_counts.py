from pathlib import Path

from src.Preprocessor import Preprocessor
from src.BagofWords import BagOfWords


def main():
    pre = Preprocessor(remove_stopwords=True, remove_names=True, remove_contractions=True)
    docs, y = pre.load_corpus(Path("chapters"))

    bow = BagOfWords(labels=("19th", "20th"))
    bow.fit(docs, y)

    for label in ("19th", "20th"):
        counts = bow.get_counts(label)
        print(f"\nTop 25 raw-count words for {label}:")
        for w, c in counts.most_common(25):
            print(w, c)


if __name__ == "__main__":
    main()
