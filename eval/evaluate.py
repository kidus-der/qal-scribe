"""Accuracy gate: character and word error rate on a folder of <name>.wav + <name>.txt (16 kHz mono).

uv run python scripts/gpu_lock.py -- uv run python eval/evaluate.py eval/data/fleurs-am-test [--no-lm] [--compare] [--limit N]
Prints one JSON line: {"clips": N, "cer": .., "wer": .., "lm": bool, "seconds": ..}. Scoring folds Amharic
homophones and drops punctuation, matching how the model authors score (see qal_scribe.textnorm).
--compare decodes every clip both greedy and with the language model from one model pass, prints the two side
by side on stderr, and adds "greedy": {"cer", "wer"} and "decode_ms": {"greedy", "lm"} (mean per clip) to the line.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

from qal_scribe import decode, models
from qal_scribe.engine import Engine
from qal_scribe.textnorm import edit_distance, normalize


def load_clips(data: Path, limit: int | None = None) -> list[tuple]:
    """(audio, normalised reference) for each <name>.wav in `data`."""
    clips = []
    for wav in sorted(data.glob("*.wav"))[:limit]:
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == 16000 and audio.ndim == 1, f"{wav}: need 16 kHz mono"
        clips.append((audio, normalize(wav.with_suffix(".txt").read_text(encoding="utf-8"))))
    if not clips:
        sys.exit(f"no .wav files in {data}")
    return clips


def error_rates(refs: list[str], hyps: list[str]) -> dict:
    char_err = char_tot = word_err = word_tot = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        hyp = normalize(hyp)
        char_err += edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        char_tot += len(ref.replace(" ", ""))
        word_err += edit_distance(ref.split(), hyp.split())
        word_tot += len(ref.split())
    return {"cer": round(char_err / char_tot, 4), "wer": round(word_err / word_tot, 4)}


def print_comparison(greedy: dict, lm: dict, ms: dict) -> None:
    print(f"{'':8}{'CER':>8}{'WER':>8}{'ms/clip':>9}", file=sys.stderr)
    for name, r in (("greedy", greedy), ("lm", lm)):
        print(f"{name:8}{r['cer']:8.2%}{r['wer']:8.2%}{ms[name]:9.1f}", file=sys.stderr)
    print(f"{'change':8}{lm['cer'] - greedy['cer']:+8.2%}{lm['wer'] - greedy['wer']:+8.2%}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--no-lm", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--language", default="amh")
    args = p.parse_args()
    if args.compare and args.no_lm:
        sys.exit("--compare needs the language model; drop --no-lm")

    clips = load_clips(args.data, args.limit)
    spec = models.get(args.language)
    engine = Engine(spec, use_lm=not args.no_lm)
    engine.load()
    if args.compare and engine.lm is None:
        sys.exit("--compare: the language model is not available (see qal-scribe doctor)")
    refs = [ref for _, ref in clips]
    variants = {"greedy": None, "lm": engine.lm}
    if not args.compare:  # the last variant is the one a normal run reports
        variants = {"lm": engine.lm} if engine.lm else {"greedy": None}
    hyps: dict[str, list[str]] = {name: [] for name in variants}
    spent = dict.fromkeys(variants, 0.0)
    t = time.perf_counter()
    for audio, _ in clips:
        logits = engine.logits(audio)
        for name, lm in variants.items():
            start = time.perf_counter()
            hyps[name].append(decode.decode(logits, engine.processor, spec, lm))
            spent[name] += time.perf_counter() - start
    line = {"clips": len(clips), **error_rates(refs, hyps[list(variants)[-1]]), "lm": engine.lm is not None,
            "seconds": round(time.perf_counter() - t, 1)}
    if args.compare:
        greedy = error_rates(refs, hyps["greedy"])
        ms = {k: 1000 * v / len(clips) for k, v in spent.items()}
        print_comparison(greedy, error_rates(refs, hyps["lm"]), ms)
        line |= {"greedy": greedy, "decode_ms": {k: round(v, 1) for k, v in ms.items()}}
    print(json.dumps(line))


if __name__ == "__main__":
    main()
