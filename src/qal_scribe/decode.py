"""CTC decoding: greedy, or a character beam search in which the model's KenLM word 5-gram scores every word.

Contract: decode(logits, processor, spec, lm=None) -> str. `logits` is a [frames, vocab] float32 numpy array
(raw model output). load_lm(model_dir, spec) returns an object with .decode(logits) -> str, or None when the
LM file or its decoder library (flashlight-text, with no wheels for Python 3.13+) is unavailable; decode() then
falls back to greedy. The first load_lm builds a word list next to the LM file (about 10 s, then cached).
ALPHA and BETA are on pyctcdecode's scale (the model's decoder.json says 0.3 and 0.5) but re-tuned on the
FLEURS Amharic dev split with eval/tune_lm.py.
split_pauses(logits, processor) -> the logits cut at the model's pauses, for decoding sentence by sentence.
"""
import json
import logging
import math
import mmap
import os
import struct
from pathlib import Path

import numpy as np

from qal_scribe.models import ModelSpec

log = logging.getLogger(__name__)

ALPHA = 0.25  # weight of the LM log-probability (natural-log units, as pyctcdecode)
BETA = 1.0  # bonus per word; offsets the LM's cost of every extra word
BEAM = 50
MIN_LOGP = -6.0  # characters rarer than this in a frame are not tried
TOP_CHARS = 8  # nor are more than this many per frame: bounds the work on noise, where the model is unsure
MIN_UNIGRAM = -7.5  # log10; the LM vocabulary is 8M words (mostly seen once), the useful ones are ~1.4M
_NEG = -1e30
_LN10 = math.log(10)
_NOT_WORDS = {"<unk>", "<s>", "</s>"}
MAX_BLOB = 8  # most greedy words one unknown LM word is checked against (FLEURS blobs join up to 6)
# Frames without a letter that end a sentence: 0.48 s at hohe-asr's 40 ms per frame. 3 dictated FLEURS sentences
# translate at chrF 53.1 cut here (0.32 s: 49.8, too many cuts), 50.1 run together, 54.5 cut at the true ends.
PAUSE_FRAMES = 12


def greedy(logits: np.ndarray, processor, spec: ModelSpec) -> str:
    text = processor.batch_decode(logits.argmax(-1)[None])[0]
    if spec.output_tag:
        text = text.replace(spec.output_tag, "")
    return " ".join(text.split())


def split_pauses(logits: np.ndarray, processor, min_frames: int = PAUSE_FRAMES) -> list[np.ndarray]:
    """Cut [frames, vocab] logits in the middle of every pause: min_frames or more frames in a row whose best
    token is no letter (a blank, word gap or tag). The model writes no ።, so pauses are where sentences end."""
    tokenizer = processor.tokenizer
    gap = getattr(tokenizer, "word_delimiter_token", "|")
    letter = np.zeros(logits.shape[1], bool)
    letter[[i for t, i in tokenizer.get_vocab().items() if len(t) == 1 and t != gap and i < len(letter)]] = True
    at = np.flatnonzero(letter[logits.argmax(-1)])
    wide = np.flatnonzero(np.diff(at) > min_frames)
    cuts = ((at[wide] + at[wide + 1]) // 2 + 1).tolist()
    return [logits[a:b] for a, b in zip([0, *cuts], [*cuts, len(logits)], strict=True)]


def _add(a: float, b: float) -> float:
    """log(exp(a) + exp(b))"""
    if a < b:
        a, b = b, a
    return a if b <= _NEG else a + math.log1p(math.exp(b - a))


class BeamLM:
    """CTC prefix beam search. A word is scored by the LM when its trailing space (or the end) is reached."""

    def __init__(self, kenlm, words, chars: dict[int, str], alpha: float = ALPHA, beta: float = BETA):
        """`words` is the flashlight Dictionary kenlm was built with; `chars` maps token id -> character."""
        self.kenlm, self.words = kenlm, words
        self.alpha, self.beta, self.beam = alpha, beta, BEAM
        self.char_ids = np.array(sorted(chars))
        self.letters = [chars[i] for i in sorted(chars)]  # same order as char_ids
        self.chars = chars
        self.unk = words.get_index("<unk>")

    def decode(self, logits: np.ndarray) -> str:
        lp = logits.astype("float32")
        lp -= lp.max(-1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
        other = np.ones(lp.shape[1], bool)  # the CTC blank and every tag or special token: all mean "no text"
        other[self.char_ids] = False
        text = self._search(lp[:, self.char_ids], np.logaddexp.reduce(lp[:, other], axis=1))
        return " ".join(self._unblob(text.split(), self._greedy(lp)))

    def _greedy(self, lp: np.ndarray) -> list[str]:
        """The greedy words: best token per frame, repeats merged, blanks and tags dropped."""
        best = lp.argmax(-1)
        best = best[np.r_[True, best[1:] != best[:-1]]] if len(best) else best
        return "".join(self.chars.get(t, "") for t in best.tolist()).split()

    def _unblob(self, words: list[str], greedy: list[str]) -> list[str]:
        """Split the blobs the search makes of names and foreign words: k words the lexicon lacks cost k <unk>
        scores, but glued together they cost one. An unknown word that is 2+ greedy words joined becomes those
        words (FLEURS test: WER 14.62% -> 14.41%, words of 12+ letters 11 -> 3)."""
        joined: dict[str, list[str]] = {}
        for i in range(len(greedy)):
            for j in range(i + 2, min(len(greedy), i + MAX_BLOB) + 1):
                joined.setdefault("".join(greedy[i:j]), greedy[i:j])
        return [part for w in words
                for part in (joined.get(w, [w]) if self.words.get_index(w) == self.unk else [w])]

    def _word(self, state, word: str):
        """LM state after `word`, and its cost in the same units as the acoustic log-probabilities."""
        state, score = self.kenlm.score(state, self.words.get_index(word))  # unknown words score as <unk>
        return state, self.alpha * _LN10 * score + self.beta

    def _search(self, chars: np.ndarray, blank: np.ndarray) -> str:
        top = min(TOP_CHARS, chars.shape[1])
        kth = np.partition(chars, -top, axis=1)[:, -top, None]  # each frame's TOP_CHARS-th best
        frame, token = np.nonzero((chars > MIN_LOGP) & (chars >= kth))
        ends = np.searchsorted(frame, np.arange(1, len(chars) + 1)).tolist()
        logp = chars[frame, token].tolist()
        text = [self.letters[i] for i in token.tolist()]
        # prefix -> [p_blank, p_nonblank, lm state, lm score]: log-probability of the paths that end in a
        # blank or in the prefix's last character, and the LM's verdict on the words completed so far.
        beams = {"": [0.0, _NEG, self.kenlm.start(False), 0.0]}
        lo, idle = 0, False
        for t, hi in enumerate(ends):
            if lo == hi and idle:
                continue  # a second blank-only frame changes every beam by the same constant
            idle, nxt = lo == hi, {}
            for prefix, (pb, pnb, state, lm) in beams.items():
                total = _add(pb, pnb)
                same = nxt.setdefault(prefix, [_NEG, _NEG, state, lm])
                same[0] = _add(same[0], total + blank[t])
                last = prefix[-1:]
                for i in range(lo, hi):
                    ch = text[i]
                    if ch == " ":
                        if last in ("", " "):  # no word to close: just another frame of the same prefix
                            same[1] = _add(same[1], total + logp[i])
                            continue
                        new_state, cost = self._word(state, prefix[prefix.rfind(" ") + 1:])
                        child = nxt.setdefault(prefix + ch, [_NEG, _NEG, new_state, lm + cost])
                        child[1] = _add(child[1], total + logp[i])
                        continue
                    if ch == last:  # a repeated character needs a blank between its two frames to count twice
                        same[1] = _add(same[1], pnb + logp[i])
                        reach = pb
                    else:
                        reach = total
                    child = nxt.setdefault(prefix + ch, [_NEG, _NEG, state, lm])
                    child[1] = _add(child[1], reach + logp[i])
            lo = hi
            if len(nxt) > self.beam:
                best = sorted(nxt.items(), key=lambda kv: -(_add(kv[1][0], kv[1][1]) + kv[1][3]))
                nxt = dict(best[:self.beam])
            beams = nxt
        return " ".join(self._best(beams).split())

    def _best(self, beams: dict) -> str:
        best, best_score = "", _NEG * 10
        for prefix, (pb, pnb, state, lm) in beams.items():
            score = _add(pb, pnb) + lm
            if prefix[-1:] not in ("", " "):
                state, cost = self._word(state, prefix[prefix.rfind(" ") + 1:])
                score += cost
            score += self.alpha * _LN10 * self.kenlm.finish(state)[1]
            if score > best_score:
                best, best_score = prefix, score
        return best


def _vocabulary(lm_path: Path) -> list[str]:
    """Every word of a KenLM binary: the file ends with the vocabulary strings, NUL-terminated, from <unk>
    (a later word may end in "<unk>", so the first match is the start). The count at byte 108 of a format-5
    header is checked, so a file laid out differently fails here instead of yielding a wrong word list."""
    with open(lm_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        start = m.find(b"<unk>\0")
        words = m[start:].decode("utf-8", errors="replace").split("\0")[:-1] if start >= 0 else []
        if len(m) < 116 or len(words) != struct.unpack_from("<Q", m, 108)[0]:
            raise ValueError(f"{lm_path} is not a KenLM binary with a vocabulary")
        return words


def _frequent(lm_path: Path, words: list[str]) -> list[str]:
    """The words whose plain unigram probability in the LM is above MIN_UNIGRAM."""
    from flashlight.lib.text.decoder.kenlm import KenLM
    from flashlight.lib.text.dictionary import Dictionary

    index = Dictionary()
    for w in words:
        index.add_entry(w)
    kenlm = KenLM(str(lm_path), index)
    nothing = kenlm.start(True)  # no <s> context
    return [w for w in words if kenlm.score(nothing, index.get_index(w))[1] > MIN_UNIGRAM]


def _lexicon(lm_path: Path) -> Path:
    """The pruned word list (one word per line) next to the LM, built from it on first use (about 10 s)."""
    path = lm_path.with_name(f"{lm_path.stem}.words{MIN_UNIGRAM:g}.txt")
    if path.exists():
        return path
    words = [w for w in _vocabulary(lm_path) if w not in _NOT_WORDS]
    keep = [w for i in range(0, len(words), 1_000_000) for w in _frequent(lm_path, words[i:i + 1_000_000])]
    keep = [w for w in keep if w.split() == [w]]  # the file holds one word per line
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # a half-written list must never look finished
    tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _chars(model_dir: Path) -> dict[int, str]:
    """Token id -> character for the model's letters; the word delimiter becomes a space."""
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    return {i: " " if t == "|" else t for t, i in vocab.items() if len(t) == 1}


def load_lm(model_dir, spec: ModelSpec):
    """Return an object with .decode(logits) -> str using spec.lm_file inside model_dir, or None when the LM
    or its decoder library is unavailable (callers then fall back to greedy)."""
    try:
        from flashlight.lib.text.decoder.kenlm import KenLM
        from flashlight.lib.text.dictionary import Dictionary
    except ImportError:
        return None
    model_dir = Path(model_dir)
    if not spec.lm_file or not (model_dir / spec.lm_file).exists():
        return None
    try:
        lm_path = model_dir / spec.lm_file
        words = Dictionary(str(_lexicon(lm_path)))
        words.add_entry("<unk>")
        words.set_default_index(words.get_index("<unk>"))
        return BeamLM(KenLM(str(lm_path), words), words, _chars(model_dir))
    except (OSError, RuntimeError, ValueError) as e:
        log.warning("language model unavailable, using greedy decoding: %s", e)
        return None


def decode(logits: np.ndarray, processor, spec: ModelSpec, lm=None) -> str:
    return greedy(logits, processor, spec) if lm is None else lm.decode(logits)
