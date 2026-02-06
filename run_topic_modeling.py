from pathlib import Path

from src.Preprocessor import Preprocessor
from src.TopicModeling import LDATopicModeler


def main():
    pre = Preprocessor(remove_stopwords=True, remove_names=True, remove_contractions=True)
    docs, labels = pre.load_corpus(Path("chapters"))

    lda = LDATopicModeler(num_topics=10)
    corpus = lda.fit(docs)

    avg = lda.average_topic_distribution(corpus, labels)

    out_dir = Path("outputs")
    lda.save_topics(out_dir, topn=25)
    lda.save_average_distributions(avg, out_dir)

    print("Saved:")
    print("  outputs/lda_topics.txt")
    print("  outputs/lda_topics.csv")
    print("  outputs/lda_avg_topic_dist.csv")

    for label in ("19th", "20th"):
        vec = avg[label]
        top = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 topics for {label}:")
        for tid, p in top:
            print("  Topic", tid, round(p, 4))


if __name__ == "__main__":
    main()
