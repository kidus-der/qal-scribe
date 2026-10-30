# Translation models

Translation (Amharic speech to English text) is optional and has a choice of model. Both are settings in
`config.toml` (`qal-scribe config --init` writes a commented file):

```toml
translation = true            # false: no translate key, the model is never downloaded or loaded
translation_model = "nllb-600m"
```

`qal-scribe --no-translate` turns translation off for one run. With translation off, `qal-scribe doctor` shows the
translation model as "off" instead of warning that it is missing, and `qal-scribe toggle --translate` says so.

## The choices

| `translation_model` | Model | Download | Licence |
|---|---|---|---|
| `nllb-600m` (default) | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 2.5 GB | CC-BY-NC-4.0 (non-commercial) |
| `nllb-1.3b` | [facebook/nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | 5.5 GB | CC-BY-NC-4.0 (non-commercial) |

Each is pinned to an exact Hugging Face revision (see `TRANSLATION_MODELS` in `src/qal_scribe/translate.py`).
Changing the setting downloads the new model on the next start (or run `qal-scribe models download --translation`).

## How they compare

Measured with `eval/translate_eval.py` on the first 200 Amharic FLEURS test clips (one row per speaker), Apple
Silicon Mac with 16 GB, PyTorch MPS, beam search of 4. "Text" translates the reference Amharic sentences; "speech"
runs the speech model (greedy, no language model) on the audio first and translates what it heard, which is what
dictation does. Scores are chrF against the English reference (higher is better, 100 is perfect). Seconds are per
sentence on the translation step alone.

| Model | Text chrF | Speech chrF | Median s | p95 s | Peak memory (process / GPU) | Load | Precision |
|---|---|---|---|---|---|---|---|
| `nllb-600m` | 57.28 | 52.11 | 0.42 | 0.72 | 2.8 GB / 2.5 GB | 9 s | float16 |
| `nllb-1.3b` | 57.95 | 54.67 | 0.70 | 1.24 | 6.9 GB / 3.4 GB | 7 s | float16 |

Reproduce one row:

```sh
uv run python eval/fetch_fleurs.py                       # the clips, once
uv run python scripts/gpu_lock.py -- uv run python eval/translate_eval.py \
    --model nllb-1.3b --speech eval/data/fleurs-am-test --asr-cache eval/data/asr.json
```

## What to pick

- **Keep the default (`nllb-600m`)** for dictation. It is the fastest and smallest, and the others do not beat it
  by the 3 chrF on speech that would justify a bigger download and more memory (`nllb-1.3b` +2.6).
- **`nllb-1.3b`** is the best step up: about 2.6 chrF better on speech, still under a second per sentence, and a
  5.5 GB download. It needs roughly 7 GB of memory while loading (the checkpoint is float32), so close other
  large apps on a 16 GB machine.
- **Not offered: MADLAD-400 (`madlad-3b`, Apache-2.0).** It scored like `nllb-1.3b` (58.15 text, 54.36 speech chrF) at
  twice the latency, but needs about 7 GB of process memory plus 6.7 GB on the GPU. On a 16 GB Mac three of four full
  runs stalled for 10 minutes or more and were killed (4-5 GB of swap), so it is left out. It is the only candidate
  that allows commercial use, so revisit it for a machine with more memory.
- **Low on memory or disk?** `translation = false` skips the model altogether and dictation is unaffected.

With only 200 sentences, treat differences of a point or so as ties. The scores come from one run each; the
`text` column uses one row per speaker, so it differs slightly from the 57.95 measured earlier for `nllb-600m` on
one row per sentence. The earlier speech figure for `nllb-600m` was 52.76 against 52.11 here; that gap is not
explained (the earlier run's output was not kept), so the table values are this run's and the two are a tie.

