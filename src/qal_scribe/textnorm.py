"""Amharic text normalisation for scoring and for the optional homophone merge."""
import re
import unicodedata

# Letters that sound the same in modern Amharic; `other` folds onto `target`, order by order. `orders` counts the
# 7 vowel orders plus, where the series has one, the labialised 8th (ሠ ሧ, ፀ ፇ).
_FAMILIES = [("ሀ", "ሐ", 7), ("ሀ", "ኀ", 7), ("ሰ", "ሠ", 8), ("አ", "ዐ", 7), ("ጸ", "ፀ", 8)]
FOLD: dict[str, str] = {chr(ord(other) + i): chr(ord(target) + i) for target, other, orders in _FAMILIES
                        for i in range(orders)}
# Labialised forms land on the spelling Amharic uses (ኋላ); ኈ ኊ ኌ ኍ have no ሀ-series twin, so they stay.
# The glottal 4th order is written አ in everyday Amharic, so ዓ and the Tigrinya-style ኣ both fold to it.
FOLD.update({"ሗ": "ኋ", "ኇ": "ሇ", "ዓ": "አ", "ኣ": "አ"})

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def fold_homophones(text: str) -> str:
    """One character in, one character out, so offsets in the folded text are valid in the original."""
    return "".join(FOLD.get(c, c) for c in text)


def normalize(text: str) -> str:
    """Scoring form: NFC, homophones folded, punctuation (incl. ። ፣ ፤) dropped, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT.sub(" ", fold_homophones(text).replace("።", " ").replace("፣", " ").replace("፤", " "))
    return " ".join(text.split())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
