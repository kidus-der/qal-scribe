"""Translation quality gate: chrF of Amharic -> English on FLEURS test sentences (which come from FLORES).

uv run python scripts/gpu_lock.py -- uv run python eval/translate_eval.py [--limit 200] [--fp32] [--device cpu]
    [--speech eval/data/fleurs-am-test]
Downloads only data/am_et/test.tsv and data/en_us/test.tsv from google/fleurs, joins them on the shared sentence
id, translates the REFERENCE Amharic text (text quality only, no speech errors) and scores against the English
reference. Prints one JSON line: {"sentences": N, "chrf": .., "median_s": .., "p95_s": .., "device": .., "dtype": ..}.
With --speech DIR (the 16 kHz FLEURS clips, named by the TSV file_name), the clips are also transcribed (no
language model) and translated, adding "chrf_speech": the full speech -> English path on the same clips.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import sacrebleu
import torch
from huggingface_hub import hf_hub_download

from qal_scribe.engine import pick_device
from qal_scribe.translate import Translator


def read_tsv(lang: str) -> list[list[str]]:
    path = hf_hub_download("google/fleurs", f"data/{lang}/test.tsv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))  # id, file, raw transcription, ...


def fleurs_items(limit: int, clips: Path | None) -> list[tuple[str, str, Path | None]]:
    """(amharic, english, wav) per sentence in file order. Several speakers share a sentence id: text-only runs
    keep one row per id, --speech runs keep every row whose clip exists in the folder."""
    english = {row[0]: row[2] for row in read_tsv("en_us")}
    rows = read_tsv("am_et")
    if clips:
        rows = [row for row in rows if (clips / row[1]).exists()]
    else:
        first: dict[str, list[str]] = {}
        for row in rows:
            first.setdefault(row[0], row)
        rows = list(first.values())
    return [(row[2], english[row[0]], clips / row[1] if clips else None) for row in rows if row[0] in english][:limit]


def timed_translate(t: Translator, texts: list[str]) -> tuple[list[str], list[float]]:
    out, times = [], []
    for text in texts:
        start = time.perf_counter()
        out.append(t.translate(text))
        times.append(time.perf_counter() - start)
    return out, times


def transcribe(clips: list[Path]) -> list[str]:
    import soundfile as sf

    from qal_scribe import models
    from qal_scribe.engine import Engine

    engine = Engine(models.get("amh"), use_lm=False)
    engine.load()
    return [engine.transcribe(sf.read(clip, dtype="float32")[0]).text for clip in clips]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--fp32", action="store_true", help="force float32 (compare against the default float16 on mps)")
    p.add_argument("--device")
    p.add_argument("--speech", type=Path, help="folder of FLEURS am_et test clips, to also score speech -> English")
    args = p.parse_args()

    items = fleurs_items(args.limit, args.speech)
    if len(items) < 100:
        sys.exit(f"only {len(items)} joined sentences; need >= 100")
    t = Translator(pick_device(args.device), torch.float32 if args.fp32 else None)
    t.load()
    t.translate(items[0][0])  # warm-up: GPU kernels compile on first use
    hyps, times = timed_translate(t, [am for am, _, _ in items])
    refs = [[en for _, en, _ in items]]
    for (am, en, _), hyp in list(zip(items, hyps, strict=True))[:3]:
        print(f"AM  {am}\nREF {en}\nHYP {hyp}\n", file=sys.stderr)
    result = {
        "sentences": len(items), "chrf": round(sacrebleu.corpus_chrf(hyps, refs).score, 2),
        "median_s": round(statistics.median(times), 3), "p95_s": round(statistics.quantiles(times, n=20)[-1], 3),
        "device": t.device, "dtype": str(t.model.dtype).removeprefix("torch."),
    }
    if args.speech:
        asr = transcribe([wav for _, _, wav in items])
        speech_hyps, speech_times = timed_translate(t, asr)
        result["chrf_speech"] = round(sacrebleu.corpus_chrf(speech_hyps, refs).score, 2)
        result["median_s_speech"] = round(statistics.median(speech_times), 3)
        for a, hyp in list(zip(asr, speech_hyps, strict=True))[:3]:
            print(f"ASR {a}\nHYP {hyp}\n", file=sys.stderr)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
