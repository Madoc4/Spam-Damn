import json
import sys
from tokens import tokenize
from pathlib import Path
from collections import defaultdict
from nb import naive_bayes


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python main.py train <positive_dir> <negative_dir> <model_prefix>"
        )
        print("   or: python main.py predict <model_prefix> <email_file>")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "train":
        if len(sys.argv) != 5:
            print(
                "Usage: python main.py train <positive_dir> <negative_dir> <model_prefix>"
            )
            sys.exit(1)
        positive_dir = sys.argv[2]
        negative_dir = sys.argv[3]
        prefix = sys.argv[4]
        train(Path(positive_dir), Path(negative_dir), prefix)

    elif cmd == "predict":
        if len(sys.argv) != 4:
            print("Usage: python main.py predict <email_file> <model_prefix>")
            sys.exit(1)
        email_file = sys.argv[2]
        prefix = sys.argv[3]
        predict(Path(email_file), prefix)

    else:
        print("Use 'train' or 'predict'.")


def train(positive_dir: Path, negative_dir: Path, prefix: str) -> None:
    positive_counts = count_dir(positive_dir)
    n_positive = len([f for f in positive_dir.iterdir() if f.is_file()])
    negative_counts = count_dir(negative_dir)
    n_negative = len([f for f in negative_dir.iterdir() if f.is_file()])

    positive_data = {"counts": positive_counts, "amt": n_positive}
    negative_data = {"counts": negative_counts, "amt": n_negative}

    with open(f"{prefix}_positive_counts.json", "w") as f:
        json.dump(positive_data, f)
    with open(f"{prefix}_negative_counts.json", "w") as f:
        json.dump(negative_data, f)

    print("Training finished!")


def predict(email_path: Path, prefix: str) -> None:
    with open(f"{prefix}_positive_counts.json") as f:
        positive_data = json.load(f)
    with open(f"{prefix}_negative_counts.json") as f:
        negative_data = json.load(f)

    with open(email_path, encoding="latin-1") as f:
        email_words = tokenize(f.read())

    result = naive_bayes(
        email_words,
        positive_data["counts"],
        negative_data["counts"],
        positive_data["amt"],
        negative_data["amt"],
    )

    print(f"Naive bayes classified as: {'positive' if result else 'negative'}")


def count_dir(dir_path: Path) -> defaultdict[str, int]:
    word_counts: defaultdict[str, int] = defaultdict(int)

    for entry in dir_path.iterdir():
        if not entry.is_file():
            continue

        with open(entry, encoding="latin-1") as f:
            for word in tokenize(f.read()):
                word_counts[word] += 1

    return word_counts


if __name__ == "__main__":
    main()
