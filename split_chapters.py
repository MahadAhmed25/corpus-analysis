import re
from pathlib import Path


P_CHAPTER = re.compile(
    r"(?im)^\s*chapter\s+(\d+|[ivxlcdm]+)\b.*$"
)

P_ROMAN_ALONE = re.compile(
    r"(?im)^\s*([ivxlcdm]{1,8})\s*$"
)

P_SECTION_WORDS = re.compile(
    r"(?im)^\s*(book|part|episode)\s+(\d+|[ivxlcdm]+)\b.*$"
)


def find_splits(text: str):
    patterns = [P_CHAPTER, P_SECTION_WORDS, P_ROMAN_ALONE]

    for pat in patterns:
        matches = list(pat.finditer(text))
        if len(matches) >= 5:
            return [m.start() for m in matches]

    return []


def split_by_indices(text: str, starts):
    chunks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        if len(chunk.split()) > 200:
            chunks.append(chunk)
    return chunks


def process_book(book_path: Path, out_dir: Path, label: str):
    text = book_path.read_text(encoding="utf-8", errors="ignore")

    starts = find_splits(text)
    if not starts:
        print(f"  !! No split pattern matched for {book_path.name}")
        return 0

    chapters = split_by_indices(text, starts)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, chapter in enumerate(chapters, start=1):
        out_file = out_dir / f"{label}_{book_path.stem}_ch{i:03d}.txt"
        out_file.write_text(chapter, encoding="utf-8")

    return len(chapters)


def main():
    for century in ["19th", "20th"]:
        in_dir = Path("data") / century
        out_dir = Path("chapters") / century

        for book in in_dir.glob("*.txt"):
            print(f"Splitting {book.name}")
            n = process_book(book, out_dir, century)
            if n:
                print(f"  -> wrote {n} sections")


if __name__ == "__main__":
    main()
