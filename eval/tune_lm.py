"""Tune the LM decoder's weight (alpha) and word bonus (beta) on a dev split; never point it at a test split.

uv run python scripts/gpu_lock.py -- uv run python eval/tune_lm.py eval/data/fleurs-am-dev
Runs the model once, then decodes the same logits for every (alpha, beta) pair and prints WER and CER.
The values in qal_scribe/decode.py are from the middle of the table's flat region, not its single best cell.
"""
import argparse
import sys
from pathlib import Path

from evaluate import error_rates, load_clips

from qal_scribe import models
from qal_scribe.engine import Engine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.15, 0.2, 0.25, 0.3, 0.35])
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    args = p.parse_args()
    if "test" in args.data.name:
        sys.exit("tune on the dev split, not on test")

    clips = load_clips(args.data)
    engine = Engine(models.get("amh"))
    engine.load()
    if engine.lm is None:
        sys.exit("the language model is not available")
    refs = [ref for _, ref in clips]
    logits = [engine.logits(audio) for audio, _ in clips]
    print("WER (CER) in %, rows alpha, columns beta")
    print(f"{'':6}" + "".join(f"{b:>16}" for b in args.betas))
    for alpha in args.alphas:
        engine.lm.alpha = alpha
        row = []
        for beta in args.betas:
            engine.lm.beta = beta
            r = error_rates(refs, [engine.lm.decode(x) for x in logits])
            row.append(f"{r['wer']:.2%} ({r['cer']:.2%})")
        print(f"{alpha:<6}" + "".join(f"{cell:>16}" for cell in row), flush=True)


if __name__ == "__main__":
    main()
