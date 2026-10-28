"""Translation quality gate: chrF of Amharic -> English on FLEURS test sentences (which come from FLORES).

uv run python scripts/gpu_lock.py -- uv run python eval/translate_eval.py [--limit 200] [--fp32] [--device cpu]
Downloads only data/am_et/test.tsv and data/en_us/test.tsv from google/fleurs, joins them on the shared sentence
id, translates the REFERENCE Amharic text (text quality only, no speech errors) and scores against the English
reference. Prints one JSON line: {"sentences": N, "chrf": .., "median_s": .., "p95_s": .., "device": .., "dtype": ..}.
"""
import argparse
import csv
import json
import statistics
import sys
import time

import sacrebleu
import torch
from huggingface_hub import hf_hub_download

from qal_scribe.engine import pick_device
from qal_scribe.translate import Translator


def fleurs_pairs(limit: int) -> list[tuple[str, str]]:
    """(amharic, english) reference transcripts, one pair per FLEURS sentence id, in file order."""
    def read(lang: str) -> dict[str, str]:
        path = hf_hub_download("google/fleurs", f"data/{lang}/test.tsv", repo_type="dataset")
        with open(path, encoding="utf-8", newline="") as f:
            rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            texts: dict[str, str] = {}
            for row in rows:  # columns: id, file, raw transcription, ...; several speakers share an id
                texts.setdefault(row[0], row[2])
            return texts

    am, en = read("am_et"), read("en_us")
    return [(am[i], en[i]) for i in am if i in en][:limit]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--fp32", action="store_true", help="force float32 (compare against the default float16 on mps)")
    p.add_argument("--device")
    args = p.parse_args()

    pairs = fleurs_pairs(args.limit)
    if len(pairs) < 100:
        sys.exit(f"only {len(pairs)} joined sentences; need >= 100")
    t = Translator(pick_device(args.device), torch.float32 if args.fp32 else None)
    t.load()
    t.translate(pairs[0][0])  # warm-up: GPU kernels compile on first use
    hyps, times = [], []
    for am, _ in pairs:
        start = time.perf_counter()
        hyps.append(t.translate(am))
        times.append(time.perf_counter() - start)
    for (am, en), hyp in list(zip(pairs, hyps, strict=True))[:3]:
        print(f"AM  {am}\nREF {en}\nHYP {hyp}\n", file=sys.stderr)
    score = sacrebleu.corpus_chrf(hyps, [[en for _, en in pairs]])
    print(json.dumps({
        "sentences": len(pairs), "chrf": round(score.score, 2),
        "median_s": round(statistics.median(times), 3), "p95_s": round(statistics.quantiles(times, n=20)[-1], 3),
        "device": t.device, "dtype": str(t.model.dtype).removeprefix("torch."),
    }))


if __name__ == "__main__":
    main()
