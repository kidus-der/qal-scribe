"""Text clean-up after transcription (workstream `amharic`).

Contract: process(text, cfg, words) -> str. Pure function, no I/O, no model. Each step is opt-in via Config:
  merge_homophones, numbers_to_digits, punctuation (experimental), plus the user's word list (always on).
Order: whitespace, word list, homophones, numbers, punctuation; whitespace is normalised again at the end.

Word list (`heard => wanted`): whole words and multi-word phrases, longest phrase first, so `ኮድ ኤዲተር` beats
`ኮድ`. Matching ignores homophone spelling (ሠላም matches a `ሰላም` entry); `wanted` may be English or empty
(deletes the word). Affixed forms (ከርሰርን) are different words and need their own entry.

Numbers (numbers_to_digits): 0-9, 10-99 (አስራ አንድ = 11, ሃያ አምስት = 25), hundreds, ሺህ, ሚሊዮን, ቢሊዮን, ትሪሊዮን
(ሶስት ሺህ አምስት መቶ = 3500), ነጥብ decimals (ሁለት ነጥብ አምስት = 2.5; the digits after ነጥብ are read one by one).
Spelling variants ሶስት/ሦስት, አስር/አሥር, ሃያ/ሀያ/ሐያ, ሚሊዮን/ሚሊየን are all accepted. A number ends where the grammar
ends, so neighbours stay separate: አንድ ሁለት = "1 2", ሃያ አስር = "20 10". A bare አስራ, a ነጥብ that is not
between a number and digit words, and ordinals (ሁለተኛ) are left alone. The prepositions በ ለ ከ የ stay attached,
as Amharic writes them (በሶስት ሰዓት = በ3 ሰዓት); በመቶ ("percent") is never read as በ + 100.
Known limit: አንድ is also the indefinite article, so "አንድ ቀን" becomes "1 ቀን"; leave the flag off if that hurts.

Punctuation (EXPERIMENTAL, rule-based, tuned to produce few wrong marks rather than many right ones):
  1. Text with no Ethiopic letter (digits, English only) is left alone.
  2. Text that already ends in punctuation is left alone, so the step is idempotent.
  3. Otherwise the last sentence (after any ። ? ! the model emitted) gets ? if it is a question, else ።.
  4. A question has an interrogative word anywhere (ምን ማን የት መቼ እንዴት ስንት የትኛው ምንድነው ማነው ወይስ ይሆን, also with
     the prefixes በ ለ ከ የ ስለ እስከ: ለምን, ከየት, የማን) or ends with the particle ወይ or the tag አይደል. ምንም, ማንም,
     የትም, መቼም ("nothing", "anyone", "anywhere", "never") are different words and do not count.
  5. Reported questions are statements: a sentence with a quotative (ብሎ ብዬ ...) or እንደሆነ/አላውቅም is not marked ?.
  6. The yes/no suffix -ን is deliberately not used: it also ends accusatives and every 1st person plural verb.
  7. A comma ፣ goes before the clause openers ነገር ግን, ሆኖም and ምክንያቱም when a word precedes them.
"""
import re

from qal_scribe.config import Config
from qal_scribe.textnorm import fold_homophones


def process(text: str, cfg: Config, words: dict[str, str] | None = None) -> str:
    text = " ".join(text.split())
    if words:
        text = replace_words(text, words)
    if cfg.merge_homophones:
        text = fold_homophones(text)
    if cfg.numbers_to_digits:
        text = numbers_to_digits(text)
    if cfg.punctuation:
        text = add_punctuation(text)
    return " ".join(text.split())


# --- word list -------------------------------------------------------------------------------------------------


def replace_words(text: str, words: dict[str, str]) -> str:
    table = {" ".join(fold_homophones(heard).split()): wanted for heard, wanted in words.items() if heard.split()}
    if not table:
        return text
    phrases = sorted(table, key=len, reverse=True)  # regex alternation takes the first that fits: longest first
    rx = re.compile(r"(?<!\w)(?:" + "|".join(r"\s+".join(map(re.escape, p.split())) for p in phrases) + r")(?!\w)")
    folded, out, last = fold_homophones(text), [], 0
    for m in rx.finditer(folded):
        out += [text[last : m.start()], table[" ".join(m.group().split())]]
        last = m.end()
    return "".join(out) + text[last:]


# --- numbers -----------------------------------------------------------------------------------------------------


def _key(word: str) -> str:
    return fold_homophones(word).replace("ሃ", "ሀ")  # ሃያ and ሀያ are both written


# kind: z zero, u 1-9, ten 10, teen "አስራ" (+ unit), tens 20-90, hun መቶ, scale ሺህ and up, point ነጥብ
_WORDS: dict[str, tuple[str, int]] = {
    _key(w): entry
    for entry, spellings in [
        (("z", 0), "ዜሮ"), (("u", 1), "አንድ"), (("u", 2), "ሁለት"), (("u", 3), "ሶስት"), (("u", 4), "አራት"),
        (("u", 5), "አምስት"), (("u", 6), "ስድስት"), (("u", 7), "ሰባት"), (("u", 8), "ስምንት"), (("u", 9), "ዘጠኝ"),
        (("ten", 10), "አስር"), (("teen", 10), "አስራ"),
        (("tens", 20), "ሃያ"), (("tens", 30), "ሰላሳ"), (("tens", 40), "አርባ"), (("tens", 50), "ሃምሳ"),
        (("tens", 60), "ስልሳ"), (("tens", 70), "ሰባ"), (("tens", 80), "ሰማንያ ሰማኒያ"), (("tens", 90), "ዘጠና"),
        (("hun", 100), "መቶ"), (("scale", 10**3), "ሺህ ሺ"), (("scale", 10**6), "ሚሊዮን ሚሊየን ሚልዮን"),
        (("scale", 10**9), "ቢሊዮን ቢሊየን ቢልዮን"), (("scale", 10**12), "ትሪሊዮን ትሪሊየን"), (("point", 0), "ነጥብ"),
    ]
    for w in spellings.split()
}
_PREFIXES = "በለከየ"
_EDGE = re.compile(r"(\W*)(\w+)(\W*)")  # punctuation, word, punctuation


def _small(run: list[tuple[str, int]], k: int) -> tuple[int, int] | None:
    """Value below 1000 at run[k:] -> (value, next k). A unit followed by መቶ always belongs to that hundred."""
    r, start, total = [*run, ("", 0), ("", 0)], k, 0  # padding keeps the look-ahead in range
    if r[k][0] == "u" and r[k + 1][0] == "hun":
        total, k = r[k][1] * 100, k + 2
    elif r[k][0] == "hun":
        total, k = 100, k + 1
    if r[k][0] == "tens":
        total, k = total + r[k][1], k + 1
        if r[k][0] == "u" and r[k + 1][0] != "hun":
            total, k = total + r[k][1], k + 1
    elif r[k][0] == "ten":
        total, k = total + 10, k + 1
    elif r[k][0] == "teen" and r[k + 1][0] == "u":
        total, k = total + 10 + r[k + 1][1], k + 2
    elif r[k][0] == "u" and r[k + 1][0] != "hun":
        total, k = total + r[k][1], k + 1
    return (total, k) if k > start else None


def _integer(run: list[tuple[str, int]]) -> tuple[int, int] | None:
    """Whole number at the start of run -> (value, tokens used). Scales must descend (ሚሊዮን before ሺህ)."""
    if run[0][0] == "z":
        return 0, 1
    total, k, below = 0, 0, float("inf")
    while True:
        small = _small(run, k)
        j = small[1] if small else k
        if j < len(run) and run[j][0] == "scale":
            if run[j][1] >= below:  # a new number starts at `small`, not a trailing part of this one
                return (total, k) if k else None
            total += (small[0] if small else 1) * run[j][1]
            k, below = j + 1, run[j][1]
            continue
        if small:
            total, k = total + small[0], j
        return (total, k) if k else None


def _parse(run: list[tuple[str, int]]) -> tuple[str, int, int] | None:
    """Number at the start of run -> (digits, tokens used, trailing tokens to keep as written)."""
    whole = _integer(run)
    if whole is None:
        return None
    value, k = whole
    if k < len(run) and run[k][0] == "point":
        j = k + 1
        while j < len(run) and run[j][0] in ("u", "z"):
            j += 1
        if j > k + 1:
            keep = int(j < len(run) and run[j][0] in ("hun", "scale"))  # 2.5 ሚሊዮን stays "2.5 ሚሊዮን"
            return f"{value}." + "".join(str(e[1]) for e in run[k + 1 : j]), j + keep, keep
    return str(value), k, 0


def _number(toks: list[str], parts: list, plain: list, i: int) -> tuple[str, int] | None:
    """Replacement for the number starting at toks[i] -> (text, index of the next token), or None."""
    if parts[i] is None:
        return None
    lead, core, _ = parts[i].groups()
    prefix, entry = "", plain[i]
    if entry is None and core[0] in _PREFIXES:
        prefix, entry = core[0], _WORDS.get(_key(core[1:]))
    if entry is None or (prefix == "በ" and entry[0] in ("hun", "scale")):  # በመቶ is "percent"
        return None
    run, j = [entry], i
    while j + 1 < len(toks) and plain[j + 1] and not parts[j][3] and not parts[j + 1][1]:  # no punctuation between
        j += 1
        run.append(plain[j])
    parsed = _parse(run)
    if parsed is None:
        return None
    digits, used, keep = parsed
    last = i + used - 1 - keep
    piece = lead + prefix + digits + parts[last].group(3)
    return (f"{piece} {toks[last + 1]}" if keep else piece), i + used


def numbers_to_digits(text: str) -> str:
    toks = text.split()
    parts = [_EDGE.fullmatch(t) for t in toks]
    plain = [_WORDS.get(_key(m[2])) if m else None for m in parts]
    out, i = [], 0
    while i < len(toks):
        found = _number(toks, parts, plain, i)
        piece, nxt = found if found else (toks[i], i + 1)
        out.append(piece)
        i = nxt
    return " ".join(out)


# --- punctuation (experimental) ---------------------------------------------------------------------------------

_WH = frozenset({
    "ምን", "ምንድን", "ምንድነው", "ምንድናቸው", "ማን", "ማንን", "ማነው", "ማናት", "ማነች", "ማናቸው", "የት", "የትኛው", "የትኛዋ",
    "የትኞቹ", "መቼ", "እንዴት", "ስንት", "ወዴት", "ወይስ", "ይሆን",
})
_WH_PREFIXES = ("በ", "ለ", "ከ", "የ", "ስለ", "እስከ")
_FINAL_PARTICLES = frozenset({"ወይ", "አይደል"})
# "I asked what you did", "I don't know who it is": the question word sits in a statement
_REPORTED = frozenset({"ብሎ", "ብላ", "ብለው", "ብዬ", "ብለህ", "ብለሽ", "ብለን", "እንደሆነ", "አላውቅም"})
_ETHIOPIC_LETTER = re.compile("[ሀ-ፚ]")  # U+1200-U+135A, the syllables
_SENTENCE_END = re.compile(r"[።?!፧…]")
_CLAUSE_OPENER = re.compile(r"(?<=\w) (?=(?:ነገር ግን|ሆኖም|ምክንያቱም)(?!\w))")


def _is_wh(word: str) -> bool:
    return word in _WH or any(word.startswith(p) and word[len(p) :] in _WH for p in _WH_PREFIXES)


def _is_question(sentence: str) -> bool:
    words = re.findall(r"\w+", sentence)
    if not words or _REPORTED.intersection(words):
        return False
    return words[-1] in _FINAL_PARTICLES or any(_is_wh(w) for w in words)


def add_punctuation(text: str) -> str:
    text = text.strip()
    if not _ETHIOPIC_LETTER.search(text) or not re.search(r"\w$", text):
        return text
    text = _CLAUSE_OPENER.sub("፣ ", text)
    return text + ("?" if _is_question(_SENTENCE_END.split(text)[-1]) else "።")
