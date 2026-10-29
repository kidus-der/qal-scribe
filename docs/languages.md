# More languages (experimental)

qal-scribe is built for Amharic. Five other languages run through the same engine with open speech models.
They are **experimental**: nobody has used them day to day, accuracy is clearly below Amharic, and there is no
language model, punctuation or text clean-up tuned for them. Expect roughly one wrong word in three to
two, and treat the output as a rough draft.

## Models

| Language | Code | Model | Licence | Download | Test sample | CER | WER | Authors' own WER |
|---|---|---|---|---|---|---|---|---|
| Tigrinya | `tir` | [badrex/Ethio-ASR-tigrinya](https://huggingface.co/badrex/Ethio-ASR-tigrinya) | CC-BY-4.0 | 2.3 GB | WAXAL test, 100 clips, 24.6 min | 17.2 % | 39.4 % | 35.65 % |
| Afaan Oromo | `orm` | [badrex/Ethio-ASR-oromo](https://huggingface.co/badrex/Ethio-ASR-oromo) | CC-BY-4.0 | 2.3 GB | WAXAL test, 100 clips, 29.1 min | 5.5 % | 22.7 % | 24.29 % |
| Afaan Oromo | `orm` | (same model) | CC-BY-4.0 | | FLEURS `om_et` test, 41 clips, 7.8 min | 18.5 % | 63.8 % | |
| Somali | `som` | [IbrahimDayax/xls-r-1b-somali-stt-asr](https://huggingface.co/IbrahimDayax/xls-r-1b-somali-stt-asr) | **CC-BY-NC-4.0 (non-commercial)** | 3.6 GB | FLEURS `so_so` test, first 100 clips, 22.2 min | 16.2 % | 44.7 % | 6.83 % (own test set) |
| Sidaama | `sid` | [badrex/Ethio-ASR-sidaama](https://huggingface.co/badrex/Ethio-ASR-sidaama) | CC-BY-4.0 | 2.3 GB | WAXAL test, 100 clips, 30.1 min | 6.1 % | 28.5 % | 30.04 % |
| Wolaytta | `wal` | [badrex/Ethio-ASR-wolaytta](https://huggingface.co/badrex/Ethio-ASR-wolaytta) | CC-BY-4.0 | 2.3 GB | WAXAL test, 100 clips, 24.9 min | 12.3 % | 38.9 % | 37.64 % |

CER and WER are error rates, so lower is better. Both are measured on the speech models as `qal-scribe` runs them
(Apple MPS, float16, greedy decoding, no language model) with `eval/evaluate.py --no-lm`. Scoring does not
lowercase, drops punctuation and collapses whitespace; the reference text was lowercased when the sample was
fetched because the models write lowercase Latin script. A 100-clip sample is small: read the numbers to within
a few points, not to the decimal.

How to read the table:

- **WAXAL** is the data these four `badrex` models were trained on. Its test split is held out, but it is the same
  kind of speech (short read sentences, similar speakers and recording conditions), so it flatters the models.
  The 100 clips are spread evenly over the test split. They come through the Hugging Face datasets-server as mp3
  and are resampled from 48 to 16 kHz, which can cost the model a little. The authors' own WER on the full test
  split is in the last column; ours agree with it within a few points.
- **FLEURS** is independent read Wikipedia-style text, so it is the fairer test. Oromo falls from 22.7 % WER on
  WAXAL to 63.8 % on FLEURS: characters are mostly right (18.5 % CER), but vowel length, doubled consonants and
  word boundaries are often wrong, and each of those makes the whole word wrong. Somali is similar.
- Tigrinya is scored without merging Amharic look-alike letters. `eval/evaluate.py` merges them (they are
  homophones in Amharic, not in Tigrinya), which gives a slightly better 16.5 % CER and 38.3 % WER on the same
  hypotheses. The table shows the stricter figure.
- The Somali authors report 6.83 % WER, but on a held-out split of their own training sources, mostly synthetic
  and YouTube-derived speech. On FLEURS the same model makes 44.7 % WER. Use ours.
- There is no FLEURS data for Tigrinya, Sidaama or Wolaytta, so those three are measured only on WAXAL and are
  likely worse on speech that differs from it.
- fp16 on MPS gives the same text as fp32 on the CPU, give or take a few characters (mostly punctuation) over
  12 to 15 clips per model, checked on all five models.

Sample data is not in the repository. Rebuild it with `eval/fetch_language_clips.py`:

```
uv run --with scipy python eval/fetch_language_clips.py fleurs om_et
uv run --with scipy python eval/fetch_language_clips.py waxal tir
uv run python scripts/gpu_lock.py -- uv run python eval/evaluate.py eval/data/waxal-tir-test --language tir --no-lm
```

## Switching language

```
qal-scribe models              # lists every language, model and licence
```

In `~/.config/qal-scribe/config.toml`:

```toml
language = "tir"    # amh (default), tir, orm, som, sid, wal
```

The first start downloads the model once (2.3 GB, Somali 3.6 GB). Hold Right Option, speak, release, as for
Amharic. `qal-scribe transcribe FILE` uses the same setting. Switching back is `language = "amh"`.

## Things that are Amharic-only

- **Translation (Right Control)** is wired for Amharic -> English. With another language selected it will not
  translate correctly. NLLB-200 itself knows Tigrinya, Oromo and Somali but not Sidaama or Wolaytta.
- **Homophone merging, number words to digits and punctuation** are written for Amharic. Leave them off for the
  other languages; merging homophones would rewrite real Tigrinya letters.
- **Language model**: none of these models ships one, so decoding is plain greedy.

## Licences

Each model's licence is on its model card and in the Hugging Face API; the links above are the cards.

- The four `badrex/Ethio-ASR-*` models are **CC-BY-4.0**: use and redistribution are allowed with credit
  (see `NOTICE`). They are fine-tuned from `facebook/w2v-bert-2.0` on the
  [WAXAL](https://huggingface.co/datasets/google/WaxalNLP) speech data. WAXAL lists the Digital Umuganda
  languages (Oromo, Sidaama, Tigrinya, Wolaytta) as CC-BY-SA-4.0. Whether a data licence reaches trained
  weights is unsettled; the authors publish the weights under CC-BY-4.0 and qal-scribe relies on that.
- `IbrahimDayax/xls-r-1b-somali-stt-asr` is **CC-BY-NC-4.0**: **non-commercial use only**. It was trained partly on
  YouTube-derived pseudo-labelled speech and on a dataset with no declared licence. That is the author's
  responsibility, but another reason to treat it as personal-use only.
- The model files are downloaded from Hugging Face on first use at a pinned revision. They are not part of the
  qal-scribe package.

## Looked at and not added

| Candidate | Why not |
|---|---|
| [badrex/Ethio-ASR-multilingual-600M](https://huggingface.co/badrex/Ethio-ASR-multilingual-600M) (CC-BY-4.0) | One model for Amharic, Tigrinya, Oromo, Sidaama and Wolaytta, but it guesses the language itself and starts its output with a tag such as `[TIR]`. The single-language models avoid wrong-language output. Not measured. |
| `badrex/Ethio-ASR-multilingual-{94M,300M,1B}` (CC-BY-NC) and `boazsew/Ethio-ASR-*` (CC-BY-NC) | Non-commercial and multilingual, with the same issues, when permissive single-language models exist. |
| [Mustafaa4a/ASR-Somali](https://huggingface.co/Mustafaa4a/ASR-Somali) (Apache-2.0) | Only a `pytorch_model.bin` (pickle) file, which the downloader does not fetch, and the Somali card above reports 35.9 % WER on its own test set against 6.83 % for the one we use. Not measured. |
| [skydheere/wav2vec2-large-mms-1b-somalia](https://huggingface.co/skydheere/wav2vec2-large-mms-1b-somalia) (CC-BY-NC) | An MMS language adapter (`target_lang`), which the engine does not load. |
| `Samuael/tigrinya-asr-characters` (Apache-2.0) | Fine-tuned from an Amharic model on Amharic data; card reports 57 % WER. |
| `misterkissi/w2v2-lg-xls-r-300m-oromo`, `DarliAI/kissi-w2v-bert-2.0-oromo`, `addisai/addis-scribe-oromo-streaming` | No licence declared, or a `.nemo` file the engine cannot load. |

No usable open CTC model with a stated licence was found for **Wolaytta or Sidaama beyond the `badrex` ones
above**; both are low-resource and WAXAL-only, so treat them as demonstrations.

## Honest quality warning

- Do not rely on these for anything where a wrong word matters: names, numbers, medical, legal.
- Tigrinya and Wolaytta get about four words in ten wrong even on the training-domain test set; Oromo and Somali
  get more than four in ten wrong on independent text.
- They are trained on read speech. Spontaneous speech, noise, other dialects and fast talkers are worse.
- Spelling follows each dataset's convention and may differ from how you write.
- Sample sizes are 41 to 100 clips. Differences of a few points between languages mean nothing.
