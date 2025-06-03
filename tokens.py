from typing import Tuple
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords


stop_words = stopwords.words("english")


def split_headers(lines: list[str]) -> Tuple[list[str], list[str]]:
    i = 0
    while lines[i].strip() != "":
        i += 1
    return lines[:i], lines[i + 1 :]


def is_html(headers: list[str]) -> bool:
    for header in headers:
        if header.startswith("Content-Type:"):
            return "text/html" in header

    return False


def tokenize(contents: str) -> list[str]:
    headers, content = split_headers(contents.splitlines())
    if is_html(headers):
        soup = BeautifulSoup("\n".join(content), features="html.parser")
        text = soup.get_text()
    else:
        text = "\n".join(content)

    final = []
    for word in word_tokenize(text.lower()):
        if word in stop_words or not word.isalpha():
            continue
        final.append(word)

    return final
