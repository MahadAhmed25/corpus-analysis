from pathlib import Path

from src.Preprocessor import Preprocessor
from gensim.corpora import Dictionary
from gensim.models import LdaModel


class LDATopicModeler:
    def __init__(self, num_topics=10):
        self.num_topics = num_topics
        self.dictionary = None
        self.model = None

    def fit(self, docs):
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=30000)
        self.dictionary.compactify()

        corpus = [self.dictionary.doc2bow(doc) for doc in docs]

        self.model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=10,
            iterations=200,
            alpha="auto",
            eta="auto",
            eval_every=None,
        )

        return corpus

    def get_topics(self, topn=25):
        topics = {}
        for tid in range(self.num_topics):
            topics[tid] = self.model.show_topic(tid, topn=topn)
        return topics
    
    ## AI GENERATED ##
    def average_topic_distribution(self, corpus, labels):
        sums = {"19th": [0.0] * self.num_topics, "20th": [0.0] * self.num_topics}
        counts = {"19th": 0, "20th": 0}

        for bow, label in zip(corpus, labels):
            counts[label] += 1
            for tid, prob in self.model.get_document_topics(bow, minimum_probability=0.0):
                sums[label][tid] += prob

        avgs = {}
        for label in ["19th", "20th"]:
            avgs[label] = [sums[label][i] / counts[label] for i in range(self.num_topics)]

        return avgs
    
    ## AI GENERATED ##
    def save_topics(self, out_dir, topn=25):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        txt_lines = []
        csv_lines = ["topic_id,rank,word,prob"]

        for tid, words in self.get_topics(topn).items():
            txt_lines.append(f"Topic {tid}")
            for rank, (word, prob) in enumerate(words, start=1):
                txt_lines.append(f"  {word}\t{prob:.4f}")
                csv_lines.append(f"{tid},{rank},{word},{prob:.6f}")
            txt_lines.append("")

        (out_dir / "lda_topics.txt").write_text("\n".join(txt_lines), encoding="utf-8")
        (out_dir / "lda_topics.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    ## AI GENERATED ##
    def save_average_distributions(self, avg, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        lines = ["label,topic_id,avg_prob"]
        for label, vec in avg.items():
            for tid, prob in enumerate(vec):
                lines.append(f"{label},{tid},{prob:.6f}")

        (out_dir / "lda_avg_topic_dist.csv").write_text("\n".join(lines), encoding="utf-8")
