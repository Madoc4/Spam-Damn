import json
import sys
from tokens import tokenize
from pathlib import Path
from collections import defaultdict
from nb import naive_bayes


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py train <spam_dir> <ham_dir> <model_prefix>")
        print("   or: python main.py predict <model_prefix> <email_file>")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "train":
        if len(sys.argv) != 5:
            print("Usage: python main.py train <spam_dir> <ham_dir> <model_prefix>")
            sys.exit(1)
        spam_dir = sys.argv[2]
        ham_dir = sys.argv[3]
        prefix = sys.argv[4]
        train(Path(spam_dir), Path(ham_dir), prefix)

    elif cmd == "predict":
        if len(sys.argv) != 4:
            print("Usage: python main.py predict <email_file> <model_prefix>")
            sys.exit(1)
        email_file = sys.argv[2]
        prefix = sys.argv[3]
        predict(Path(email_file), prefix)

    else:
        print("Use 'train' or 'predict'.")


def train(spam_dir: Path, ham_dir: Path, prefix: str) -> None:
    spam_counts = count_dir(spam_dir)
    n_spam = len([f for f in spam_dir.iterdir() if f.is_file()])
    ham_counts = count_dir(ham_dir)
    n_ham = len([f for f in ham_dir.iterdir() if f.is_file()])

    spam_data = {"counts": spam_counts, "amt": n_spam}
    ham_data = {"counts": ham_counts, "amt": n_ham}

    with open(f"{prefix}_spam_counts.json", "w") as f:
        json.dump(spam_data, f)
    with open(f"{prefix}_ham_counts.json", "w") as f:
        json.dump(ham_data, f)

    print("Training finished!")


def predict(email_path: Path, prefix: str) -> None:
    with open(f"{prefix}_spam_counts.json") as f:
        spam_data = json.load(f)
    with open(f"{prefix}_ham_counts.json") as f:
        ham_data = json.load(f)

    with open(email_path, encoding="latin-1") as f:
        email_words = tokenize(f.read())

    result = naive_bayes(
        email_words,
        spam_data["counts"],
        ham_data["counts"],
        spam_data["amt"],
        ham_data["amt"],
    )
    print(f"Naive bayes classified as: {result}")


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
