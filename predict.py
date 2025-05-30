import sys
import json
from count import sanitize
from nb import naive_bayes

def load_model(prefix):
    with open(f"{prefix}_spam_counts.json", "r") as f:
        spam_word_counts = json.load(f)
    with open(f"{prefix}_ham_counts.json", "r") as f:
        ham_word_counts = json.load(f)
    with open(f"{prefix}_meta.json", "r") as f:
        meta = json.load(f)
    return spam_word_counts, ham_word_counts, meta["n_spam"], meta["n_ham"]

def predict_email(email_path, spam_word_counts, ham_word_counts, n_spam, n_ham):
    with open(email_path) as f:
        lines = f.readlines()
    email_words = sanitize(lines)
    vocab = set(list(spam_word_counts.keys()) + list(ham_word_counts.keys()))
    return naive_bayes(email_words, spam_word_counts, ham_word_counts, n_spam, n_ham,
                       vocab)

def run(model_prefix, email_file):
    spam_word_counts, ham_word_counts, n_spam, n_ham = load_model(prefix)
    prediction = predict_email(email_path, spam_word_counts, ham_word_counts, n_spam, n_ham)
    return prediction

