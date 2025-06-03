import json
import argparse
import sys
from typing import Any, Optional
from tokens import tokenize
from pathlib import Path
from collections import defaultdict
from nb import naive_bayes


def main():
    parser = argparse.ArgumentParser(
        prog="spam-damn",
        description="Predict whether an email is of a certain class using a Naive Bayes classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("prefix", help="Prefix for saving the model")
    train_parser.add_argument(
        "--positive", help="Directory with positive training data"
    )
    train_parser.add_argument(
        "--negative", help="Directory with negative training data"
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Predict using a trained model"
    )
    predict_parser.add_argument("prefix", help="Prefix of the trained model to use")
    predict_parser.add_argument("file", help="File to make predictions on")

    args = parser.parse_args()

    if args.command == "train":
        train(
            None if args.positive is None else Path(args.positive),
            None if args.negative is None else Path(args.negative),
            args.prefix,
        )

    elif args.command == "predict":
        predict(Path(args.file), args.prefix)

    else:
        print("Use 'train' or 'predict'.")


def train(
    positive_dir: Optional[Path], negative_dir: Optional[Path], prefix: str
) -> None:
    if positive_dir is None and negative_dir is None:
        print("Must specifiy either a positive dir and/or a negative dir for training.")
        sys.exit(1)

    existing_positive_path = Path(f"{prefix}_positive_counts.json")
    if not existing_positive_path.exists():
        existing_positive = {"amt": 0, "counts": {}}
    else:
        existing_positive = json.loads(existing_positive_path.read_text())
    existing_negative_path = Path(f"{prefix}_negative_counts.json")
    if not existing_negative_path.exists():
        existing_negative = {"amt": 0, "counts": {}}
    else:
        existing_negative = json.loads(existing_negative_path.read_text())

    if negative_dir is None:
        assert positive_dir is not None
        train_class(positive_dir, "positive", existing_positive, prefix)
        print("Incremental training for negative class finished!")
        return
    elif positive_dir is None:
        assert negative_dir is not None
        train_class(negative_dir, "negative", existing_negative, prefix)
        print("Incremental training for negative class finished!")
        return

    assert positive_dir is not None and negative_dir is not None

    train_class(positive_dir, "positive", existing_positive, prefix)
    train_class(negative_dir, "negative", existing_negative, prefix)

    print("Training finished!")


def train_class(dir: Path, class_name: str, existing: Any, prefix: str):
    counts = count_dir(dir)
    n_positive = len([f for f in dir.iterdir() if f.is_file()])

    existing["amt"] += n_positive
    existing["counts"] = merge(existing["counts"], counts)

    with open(f"{prefix}_{class_name}_counts.json", "w") as f:
        json.dump(existing, f)


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


def merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    new = {}
    keys = set(a) | set(b)
    for key in keys:
        new[key] = a.get(key, 0) + b.get(key, 0)
    return new


if __name__ == "__main__":
    main()
