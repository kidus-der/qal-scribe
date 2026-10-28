"""Amharic text normalisation for scoring and for the optional homophone merge."""
import re
import unicodedata

# Letters that sound the same in modern Amharic; each family maps onto its first member, all 7 orders.
_FAMILIES = [("ሀ", "ሐ"), ("ሀ", "ኀ"), ("ሰ", "ሠ"), ("አ", "ዐ"), ("ጸ", "ፀ")]
FOLD: dict[str, str] = {}
for target, other in _FAMILIES:
    for order in range(7):
        FOLD[chr(ord(other) + order)] = chr(ord(target) + order)
FOLD.update({"ኈ": "ሆ", "ሗ": "ኋ"})

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def fold_homophones(text: str) -> str:
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
