from pathlib import Path
from src.Preprocessor import Preprocessor

def main():
    pre = Preprocessor()
    base = Path("chapters")

    for label in ["19th", "20th"]:
        files = list((base / label).glob("*.txt"))
        n_docs = len(files)

        token_counts = []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            tokens = pre.tokenize(text)
            token_counts.append(len(tokens))

        avg_tokens = sum(token_counts) / n_docs if n_docs else 0

        print(f"{label}:")
        print(f"  documents = {n_docs}")
        print(f"  avg_tokens_per_doc = {avg_tokens:.2f}")
        print()

if __name__ == "__main__":
    main()
