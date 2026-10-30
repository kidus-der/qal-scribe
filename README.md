# qal-scribe

On-device push-to-talk dictation for Amharic. Hold a key, speak, let go: the text appears wherever your cursor is.
A second key turns Amharic speech into English text. Everything runs on your computer; after the models are
downloaded nothing leaves it.

- **macOS** on Apple Silicon is the main platform (Intel Macs are not supported: PyTorch no longer builds for
  them). **Linux** works and is in beta.
- Speech model: [snapwre/hohe-asr-amharic](https://huggingface.co/snapwre/hohe-asr-amharic) (wav2vec2-BERT 600M,
  CC-BY-4.0), decoded with its KenLM language model. On an Apple GPU a 10 s clip takes about 0.15 s.
- Translation (optional, can be turned off): [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
  by default (CC-BY-NC-4.0: **non-commercial use only**); larger models are a setting, see
  [docs/translation.md](docs/translation.md).

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv tool install --python 3.12 qal-scribe
qal-scribe doctor      # checks permissions, microphone and models, and says how to fix what is missing
qal-scribe             # hold Right Option, speak, release
```

The first start downloads the speech model and its language model (about 3 GB), then the translation model
(2.5 GB); later starts work offline. Ctrl+C cancels a download (the next start begins it again). To skip the
translation model, set `translation = false` in the config, or start with `qal-scribe --no-translate`. `qal-scribe models download
[--translation]` fetches the models without starting.

On Linux without an NVIDIA GPU, install the CPU build of PyTorch instead of the multi-GB CUDA one:

```sh
uv tool install --python 3.12 qal-scribe --torch-backend cpu
```

uv does not remember `--torch-backend`, so upgrade with the same line plus `--upgrade` rather than with
`uv tool upgrade`, which may fetch the multi-GB CUDA build.

`--python 3.12` matters: the language-model decoder has no builds for newer Pythons, and without it qal-scribe
falls back to slightly less accurate greedy decoding (`qal-scribe doctor` warns about it). uv downloads Python 3.12
if you do not have it.

### macOS permissions

Give the terminal app you run qal-scribe from (Terminal, iTerm2, Ghostty, ...) these three permissions in
System Settings > Privacy & Security, then quit and reopen the terminal:

- **Microphone**, to hear you (macOS asks the first time you record).
- **Input Monitoring**, so the hotkeys work while another app is in front.
- **Accessibility**, to paste into that app.

`qal-scribe doctor` shows which ones are missing.

## Use

| Key | What it does |
|---|---|
| Hold **Right Option**, speak, release | Amharic text is pasted at your cursor |
| Hold **Right Control**, speak, release | The English translation is pasted |

Translation goes sentence by sentence, and the speech model writes no punctuation, so qal-scribe takes a pause of
half a second as the end of a sentence: pause briefly between sentences when you translate.

Pasting goes through the clipboard: qal-scribe saves what you had copied and puts it back half a second later.
While you speak, a live draft appears in the window (turn it off with `preview = false`).

The terminal window shows the state (ready, recording, transcribing), a mic level meter and a history of what
you said. Its keys: `c` copies the last text, `Enter` copies the highlighted history row, `l` switches language,
`t` changes the theme, `m` toggles a compact 3-5 line layout for a small always-visible window, `q` quits.
`qal-scribe --plain` prints one line per event instead, for logs or terminals without a full UI.

### Linux (beta)

Text is typed with `wtype` (Wayland: sway, Hyprland, ...), `xdotool` (X11) or, on GNOME and KDE Wayland,
pasted with `wl-copy` + `ydotool` (needs `ydotoold` running; terminals want Ctrl+Shift+V, so pasting into a
terminal does not work there). Without any of them the text goes to the clipboard and the window says so.
`qal-scribe doctor` tells you what to install.

Before `uv tool install`, install PortAudio and the tools to build `evdev` (a keyboard library that has no
ready-made build for Linux):

```sh
sudo apt install libportaudio2 build-essential python3-dev      # Debian, Ubuntu
sudo dnf install portaudio gcc python3-devel kernel-headers      # Fedora
sudo pacman -S portaudio base-devel                             # Arch
```

Wayland does not let programs read global hotkeys, so bind two keys in your compositor or desktop settings to:

```sh
qal-scribe toggle              # first press starts recording, second press pastes Amharic
qal-scribe toggle --translate  # the same, pasting English
```

Linux has only been tested in CI and with simulated tools so far. Please report what works and what does not.

## Settings

`qal-scribe config --init` writes a fully commented `~/.config/qal-scribe/config.toml` (or under
`$XDG_CONFIG_HOME`) listing every setting with its default; `qal-scribe config` prints its path. The ones you are
most likely to change:

| Setting | Default | |
|---|---|---|
| `dictate_key`, `translate_key` | `"alt_r"`, `"ctrl_r"` | [pynput key names](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key) such as `cmd_r`, `shift_r`, `f13`; `translate_key = ""` turns translation off |
| `translation` | `true` | `false` turns translation off: no translate key, and the translation model is never downloaded or loaded (for small machines) |
| `translation_model` | `"nllb-600m"` | `"nllb-1.3b"` or `"madlad-3b"` instead; see [docs/translation.md](docs/translation.md) for the measured differences |
| `mic` | system default | part of a microphone's name, or its number |
| `numbers_to_digits` | `false` | write spoken numbers as digits: ሁለት ሺህ አስራ ሰባት → 2017 |
| `merge_homophones` | `false` | spell sound-alike letters one way (ሐ ኀ → ሀ, ሠ → ሰ, ዐ ዓ → አ, ፀ → ጸ) |
| `punctuation` | `false` | add ። ? and ፣ by rule (**experimental**, see below) |
| `lm` | `true` | decode with the language model; `false` is plain greedy decoding |
| `sounds` | `true` | short sound when recording starts and stops |

Run `qal-scribe doctor` after editing to check the file.

### Your own words

Put corrections in `~/.config/qal-scribe/words.txt`, one per line: `heard => wanted`, for example
`ኤፒአይ => API`. Lines starting with `#` are ignored. The `wanted` text still goes through
`merge_homophones` and `numbers_to_digits` when those are on, so a wanted word containing ዓ or a number word can
change; turn those settings off if that happens.

### Punctuation (experimental)

Amharic speech models write no punctuation. With `punctuation = true`, qal-scribe adds it with simple rules: ?
after a sentence with a question word (ምን, ማን, የት, መቼ, እንዴት, ስንት, ...), ። otherwise, and ፣ before ነገር ግን, ሆኖም
and ምክንያቱም. The rules prefer a missing mark to a wrong one, but they do make mistakes: an exclamation such as
እንዴት ደስ ይላል gets ?, and a yes/no question without a question word (ዛሬ ትመጣለህ) gets ።. Check the text before
you send it.

Spoken numbers have known traps too: አንድ ("a", "one") becomes 1 in አንድ ቀን, and the city አርባ ምንጭ becomes 40 ምንጭ.

## Accuracy

Measured on the 516 clips of the FLEURS Amharic test split with `eval/evaluate.py` (lower is better; scoring folds
homophone letters and drops punctuation):

| Decoding | CER | WER |
|---|---|---|
| greedy (the model alone) | 6.5 % | 17.2 % |
| with the language model (default) | 6.0 % | 14.4 % |
| language model and `numbers_to_digits = true` | 4.5 % | 13.1 % |

The language model removes about one word error in six and adds about 11 ms per clip. The FLEURS transcripts
write numbers as digits, which is why `numbers_to_digits` scores better here; it stays off by default because of
the traps described above. These numbers are for read speech; expect more errors with background noise, fast
speech, names and English words (add those to your words file).

## More languages (experimental)

Tigrinya (`tir`), Afaan Oromo (`orm`), Somali (`som`), Sidaama (`sid`) and Wolaytta (`wal`) run through the same
engine with other open models. They are rough drafts at best: see [docs/languages.md](https://github.com/kidus-der/qal-scribe/blob/main/docs/languages.md) for
their accuracy and licences (the Somali model is non-commercial). Download one with
`qal-scribe models download tir`, then press `l` in the window or set `language = "tir"`. Translation was only
measured from Amharic (NLLB has no Sidaama or Wolaytta), and the text clean-up settings are made for Amharic.

## Commands

```
qal-scribe [--no-translate]    start dictating (terminal UI); --no-translate: dictation only
qal-scribe --plain             the same, one line per event
qal-scribe doctor              check permissions, microphone, device, config and models
qal-scribe models              list languages and whether each model is downloaded
qal-scribe models download [LANG] [--translation]
qal-scribe transcribe FILE [--translate]       an audio file (wav, flac, ogg, mp3; convert m4a first) to text
qal-scribe config [--init]     show (or create) the config file
qal-scribe toggle [--translate]                start/stop recording in the running instance
qal-scribe about               licences and model attribution
```

## Licence

The code is MIT licensed. The models are downloaded at run time and keep their own licences: the Amharic speech
model is CC-BY-4.0 and the NLLB translation models (`nllb-600m`, `nllb-1.3b`) are CC-BY-NC-4.0, which allows
**non-commercial use only**; MADLAD-400 (`madlad-3b`) is Apache-2.0.
`qal-scribe about` lists them all.

## Development

```sh
uv sync
uv run ruff check
uv run pytest                  # no model weights are loaded
uv run python scripts/gpu_lock.py -- uv run pytest -m model     # tests with the real models
uv run python eval/fetch_fleurs.py && \
  uv run python scripts/gpu_lock.py -- uv run python eval/evaluate.py eval/data/fleurs-am-test --compare
```
