import json
import sys
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from os import listdir, path


stop_words = stopwords.words("english")


def split_headers(lines):
    i = 0
    while lines[i].strip() != "":
        i += 1
    return (lines[:i], lines[i + 1 :])


def is_html(headers):
    for header in headers:
        if header.startswith("Content-Type:"):
            return "text/html" in header

    return False


def sanitize(lines):
    headers, content = split_headers(lines)
    if is_html(headers):
        soup = BeautifulSoup("\n".join(content), features="html.parser")
        content = soup.get_text()
    else:
        content = "\n".join(content)

    final = []
    for word in word_tokenize(content.lower()):
        if word in stop_words or not word.isalpha():
            continue
        final.append(word)

    return final


word_counts = defaultdict(int)
dir = sys.argv[1]
for entry in listdir(dir):
    with open(path.join(dir, entry), encoding="latin-1") as f:
        lines = f.readlines()

    for word in sanitize(lines):
        word_counts[word] += 1

with open(f"{dir}-counts.json", "w") as f:
    json.dump(word_counts, f)
