import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Common names to be removed from the text during preprocessing
COMMON_NAMES = {
    "elizabeth", "darcy", "jane", "mary", "anne", "marilla",
    "daisy", "gatsby", "tom", "lucy", "peter", "wendy",
    "john", "michael", "dorothy", "colin", "felix",
    "ahab", "queequeg", "starbuck", "stubb",
    "alice", "queen", "king", "whale", "bennet", "whales", "bingley",
    "wickham", "collins", "lydia", "diana", "matthew", "cecil", "bartlett",
    "lizzy", "dick", "moby", "oz", "lynde", "ben", "freddy", "barry",
}

CONTRACTION_FRAGMENTS = {
    "t", "ll", "ve", "re", "m", "d", "s",
    "don", "tha", "th", "didn", "isn", "wasn",
    "aren", "weren", "won", "shouldn", "couldn", "wouldn",
}


class Preprocessor:
    TOKEN_PATTERN = re.compile(r"[a-z]+")

    def __init__(self, remove_stopwords=False, remove_names=False, remove_contractions=False):
        self.remove_stopwords = remove_stopwords
        self.remove_names = remove_names
        self.remove_contractions = remove_contractions

    def tokenize(self, text):
        tokens = self.TOKEN_PATTERN.findall(text.lower())
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        if self.remove_names:
            tokens = [t for t in tokens if t not in COMMON_NAMES]
        if self.remove_contractions:
            tokens = [t for t in tokens if t not in CONTRACTION_FRAGMENTS]
        return tokens

    def load_corpus(self, base_dir, labels=("19th", "20th")):
        docs = []
        y = []

        for label in labels:
            for path in (base_dir / label).glob("*.txt"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                docs.append(self.tokenize(text))
                y.append(label)

        return docs, y
