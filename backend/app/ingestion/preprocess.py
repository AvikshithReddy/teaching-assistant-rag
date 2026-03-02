# app/ingestion/preprocess.py
from __future__ import annotations

import re
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# keep words + some structure tokens used in tables/code
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[|=><()\[\]{}%$#@:/\.-]", re.UNICODE)


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(text)
    out: List[str] = []
    for t in tokens:
        t_l = t.lower()
        # keep separators and symbols as-is
        if len(t_l) == 1 and not t_l.isalnum():
            out.append(t_l)
            continue
        if t_l in STOP_WORDS:
            continue
        out.append(LEMMATIZER.lemmatize(t_l))
    return out


def preprocess_for_tfidf(text: str) -> str:
    text = normalize_text(text)
    return " ".join(tokenize(text))