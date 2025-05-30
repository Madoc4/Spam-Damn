from count import sanitize, split_headers, is_html
from os import listdir, path
from collections import defaultdict
import json
import sys

def count_words_in_dir(directory):
    word_counts = defaultdict(int)
    email_count = 0
    for entry in listdir(directory):
        file_path = path.join(directory, entry)
        if path.isfile(file_path):
            email_count += 1
            with open(file_path, encoding="latin-1") as f:
                lines = f.readlines()
            for word in sanitize(lines):  # use sanitize from count. 
                    word_counts[word] += 1
        return word_counts, email_count

def train(spam_dir, ham_dir, output):
    spam_word_counts, n_spam = count_words_in_dir(spam_dir)
    ham_word_counts, n_ham = count_words_in_dir(ham_dir)

    with open(f"{output}_spam_counts.json", "w") as f:
        json.dump(spam_word_counts, f)

    with open(f"{output}_ham_counts.json", "w") as f:
        json.dump(ham_word_counts, f)

    meta = {"n_spam": n_spam, "n_ham": n_ham}
    with open(f"{output}_meta.json", "w"} as f:
        json.dump(meta, f)
