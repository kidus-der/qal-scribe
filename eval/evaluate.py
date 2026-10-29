"""Accuracy gate: character and word error rate on a folder of <name>.wav + <name>.txt (16 kHz mono).

uv run python scripts/gpu_lock.py -- uv run python eval/evaluate.py eval/data/fleurs-am-test [--no-lm] [--limit N]
Prints one JSON line: {"clips": N, "cer": .., "wer": .., "lm": bool, "seconds": ..}. Scoring folds Amharic
homophones (ዓ and ኣ fold to አ, ሠ to ሰ, ...) and drops punctuation, matching how the model authors score (see
qal_scribe.textnorm). Scores measured before the ዓ/ኣ fold was added are not comparable: re-measure baseline and
candidate with the same normalize().
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

from qal_scribe import models
from qal_scribe.engine import Engine
from qal_scribe.textnorm import edit_distance, normalize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--no-lm", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--language", default="amh")
    args = p.parse_args()

    clips = sorted(args.data.glob("*.wav"))[: args.limit]
    if not clips:
        sys.exit(f"no .wav files in {args.data}")
    engine = Engine(models.get(args.language), use_lm=not args.no_lm)
    engine.load()
    char_err = char_tot = word_err = word_tot = 0
    t = time.perf_counter()
    for wav in clips:
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == 16000 and audio.ndim == 1, f"{wav}: need 16 kHz mono"
        ref = normalize(wav.with_suffix(".txt").read_text(encoding="utf-8"))
        hyp = normalize(engine.transcribe(audio).text)
        char_err += edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        char_tot += len(ref.replace(" ", ""))
        word_err += edit_distance(ref.split(), hyp.split())
        word_tot += len(ref.split())
    print(json.dumps({"clips": len(clips), "cer": round(char_err / char_tot, 4), "wer": round(word_err / word_tot, 4),
                      "lm": engine.lm is not None, "seconds": round(time.perf_counter() - t, 1)}))


if __name__ == "__main__":
    main()
