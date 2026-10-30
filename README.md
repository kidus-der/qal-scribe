<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/kidus-der/qal-scribe/main/docs/assets/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/kidus-der/qal-scribe/main/docs/assets/banner.svg">
    <img alt="qal-scribe: hold a key, speak Amharic, let go, and it types at your cursor" src="https://raw.githubusercontent.com/kidus-der/qal-scribe/main/docs/assets/banner.svg" width="820">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/qal-scribe/"><img alt="PyPI" src="https://img.shields.io/pypi/v/qal-scribe?color=1d7a4f"></a>
  <a href="https://github.com/kidus-der/qal-scribe/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/kidus-der/qal-scribe/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-1d7a4f">
  <a href="https://github.com/kidus-der/qal-scribe/blob/main/LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-1d7a4f"></a>
</p>

**qal-scribe** (ቃል, *qal*, means "word") is push-to-talk dictation for Amharic that runs entirely on your
computer. Hold a key, speak, let go: the text appears wherever your cursor is, in any app. A second key turns
Amharic speech into English text.

<p align="center">
  <img alt="The qal-scribe window: setup, ready, recording with a live draft, history, a translation and the settings" src="https://raw.githubusercontent.com/kidus-der/qal-scribe/main/docs/assets/demo.svg" width="820">
</p>

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a Mac with Apple Silicon (Linux works too, in beta).

```sh
uv tool install --python 3.12 qal-scribe
qal-scribe
```

The first start opens a setup screen that walks you through the permissions and downloads the models
(about 3 GB, plus 2.5 GB if you want translation). After that it works offline. When a new version is out,
qal-scribe tells you; `qal-scribe update` installs it. The full guide, including
Linux, is in [docs/install.md](https://github.com/kidus-der/qal-scribe/blob/main/docs/install.md).

## What it does

- **Dictate Amharic.** Hold **Right Option**, speak, release. The Amharic text is pasted where you are typing.
- **Translate to English.** Hold **Right Command** (Linux: **Right Control**) instead, and the English
  translation is pasted. Translation is optional: turn it off and the translation model is never downloaded.
- **See it as you speak.** The window shows a live draft while you talk, a microphone level meter, and a
  history of everything you said, ready to copy again.
- **Five more languages, experimental.** Tigrinya, Afaan Oromo, Somali, Sidaama and Wolaytta run on the same
  engine as rough drafts ([docs/languages.md](https://github.com/kidus-der/qal-scribe/blob/main/docs/languages.md)).
- **Private.** Your voice never leaves your computer. The only network use is downloading the models.

## First run

1. Run `qal-scribe`. The setup screen opens.
2. **macOS:** turn on three permissions for the terminal app you started it from (Terminal, iTerm2, Ghostty, ...):
   **Microphone**, **Input Monitoring** (so the keys work while another app is in front) and **Accessibility**
   (so it can paste). Each row has an **Open Settings** button. Then quit the terminal with ⌘Q and open it again.
   **Linux:** the screen shows the exact install command for anything missing.
3. Choose whether you want **translation to English**, and which model.
4. Press **Download**. Each model gets its own progress bar.
5. Press **Continue**, click into any app, hold **Right Option** and speak.

Something not working? `qal-scribe doctor` checks everything and prints the fix for each problem.
[docs/permissions.md](https://github.com/kidus-der/qal-scribe/blob/main/docs/permissions.md) explains each permission.

## Keys

| Key | What it does |
|---|---|
| Hold **Right Option**, speak, release | paste the Amharic text at your cursor |
| Hold **Right Command** (Linux: **Right Control**) | paste the English translation |
| `c` | copy the last text |
| `a` | copy the Amharic behind a translation |
| `Enter` | copy the highlighted history row (`Esc` jumps back to the latest) |
| `s` | settings |
| `d` | setup: permissions and downloads |
| `l` | switch language (when you have downloaded more than one) |
| `t` / `m` | next colour theme / compact layout for a small window |
| `v` | mute or unmute the sounds (saved to your config) |
| `?` | every key, as you configured them |
| `q` | quit |

Both hotkeys can be changed in the settings (`s`). The other keys work in the qal-scribe window.

<details>
<summary><b>Settings</b></summary>

Press `s` in the window to change the common settings; each one saves at once. Everything else lives in
`config.toml`: `qal-scribe config --init` writes a commented copy and `qal-scribe config` prints where it is.

| Setting | Default | What it does |
|---|---|---|
| `translation` | `true` | `false`: no translate key, and the translation model is never downloaded or loaded |
| `dictate_key`, `translate_key` | `"alt_r"`, `"cmd_r"` (Linux `"ctrl_r"`) | the push-to-talk keys |
| `mic` | system default | part of a microphone's name, or its number |
| `numbers_to_digits` | `false` | write spoken numbers as digits: ሁለት ሺህ አስራ ሰባት → 2017 |
| `merge_homophones` | `false` | spell sound-alike letters one way (ሐ ኀ → ሀ, ሠ → ሰ, ዐ ዓ → አ, ፀ → ጸ) |
| `punctuation` | `false` | add ። ? and ፣ by rule (experimental) |
| `preview` | `true` | a live draft while you speak |
| `indicator` | `true` | macOS: a small floating pill, "Listening" while you hold the key and "Transcribing…" after, even when the terminal is hidden; ignored on Linux |

Every setting, with its default, is in [docs/settings.md](https://github.com/kidus-der/qal-scribe/blob/main/docs/settings.md),
including your own word corrections (`words.txt`) and how the experimental punctuation behaves.

</details>

<details>
<summary><b>Translation models</b></summary>

Translation goes sentence by sentence, and the speech model writes no punctuation, so qal-scribe takes a
pause of half a second as the end of a sentence: pause briefly between sentences when you translate.

Pick the model on the setup screen, in the settings, or with `translation_model` in `config.toml`:

| `translation_model` | Download | Licence |
|---|---|---|
| `nllb-600m` (default) | 2.5 GB | CC-BY-NC-4.0, non-commercial use only |
| `nllb-1.3b` | 5.5 GB | CC-BY-NC-4.0, non-commercial use only |

Bigger models are slower and need more memory. On a machine with little memory, turn translation off: the
translation model is then never loaded.

</details>

<details>
<summary><b>Linux (beta)</b></summary>

Before installing, get PortAudio and the tools to build `evdev` (a keyboard library with no ready-made build):

```sh
sudo apt install libportaudio2 build-essential python3-dev      # Debian, Ubuntu
sudo dnf install portaudio gcc python3-devel kernel-headers      # Fedora
sudo pacman -S portaudio base-devel                             # Arch
```

Without an NVIDIA GPU, install the CPU build of PyTorch instead of the multi-GB CUDA one, and upgrade with the
same line plus `--upgrade` (uv does not remember `--torch-backend`):

```sh
uv tool install --python 3.12 qal-scribe --torch-backend cpu
```

Text is typed with `wtype` (sway, Hyprland), `xdotool` (X11), or on GNOME and KDE Wayland pasted with `wl-copy`
plus `ydotool` (needs `ydotoold` running). Without any of them the text goes to the clipboard and the window
says so. Wayland gives programs no global hotkeys, so bind two keys in your desktop settings to
`qal-scribe toggle` and `qal-scribe toggle --translate`.

Linux has only been tested in CI and with simulated tools so far. Please
[report](https://github.com/kidus-der/qal-scribe/issues) what works and what does not.

</details>

<details>
<summary><b>Troubleshooting</b></summary>

Start with `qal-scribe doctor`: it checks the permissions, microphone, compute device, config and models,
and prints the fix next to each problem.

- **The keys do nothing while another app is in front.** Turn on Input Monitoring for your terminal, then quit
  it with ⌘Q and reopen it.
- **The text is not pasted.** Turn on Accessibility for your terminal. The text is also on the history list:
  select it and press `Enter` to copy it.
- **It pastes what you had copied before.** Some apps paste slowly; raise `restore_delay` (default `0.5` seconds).
- **The language model is off.** `qal-scribe doctor` warns when Python is not 3.12: reinstall with
  `uv tool install --python 3.12 --reinstall qal-scribe`.
- **A download stopped.** Ctrl+C cancels a download; the next start, `qal-scribe setup` or
  `qal-scribe models download --translation` begins it again.

More answers in [docs/faq.md](https://github.com/kidus-der/qal-scribe/blob/main/docs/faq.md).

</details>

<details>
<summary><b>Privacy</b></summary>

Recording, transcription and translation all happen on your computer. qal-scribe sends no audio, no text and
no usage data anywhere. It goes online only to fetch the models from Hugging Face (and to look up their download
sizes on the setup screen), the first time or when you choose a new one. Pasting uses the clipboard: qal-scribe saves what you had copied and puts it back half a
second later.

</details>

<details>
<summary><b>Accuracy</b></summary>

Measured on the 516 clips of the FLEURS Amharic test split with `eval/evaluate.py` (lower is better; scoring
folds homophone letters and drops punctuation):

| Decoding | CER | WER |
|---|---|---|
| greedy (the model alone) | 6.5 % | 17.2 % |
| with the language model (default) | 6.0 % | 14.4 % |
| language model and `numbers_to_digits = true` | 4.5 % | 13.1 % |

These are for read speech; expect more errors with background noise, fast speech, names and English words
(add those to your words file). On an Apple GPU a 10 s clip takes about 0.15 s.

</details>

<details>
<summary><b>Commands</b></summary>

```
qal-scribe                     start dictating (terminal UI)
qal-scribe --plain             the same, one line per event
qal-scribe --no-translate      start without the translation model, for this run
qal-scribe setup               permissions, translation choice and model downloads
qal-scribe doctor              check permissions, microphone, device, config and models
qal-scribe models              list languages and whether each model is downloaded
qal-scribe models download [LANG] [--translation]
qal-scribe transcribe FILE [--translate]       an audio file (wav, flac, ogg, mp3; convert m4a first) to text
qal-scribe config [--init]     show (or create) the config file
qal-scribe toggle [--translate]                start/stop recording in the running instance
qal-scribe about               licences and model attribution
qal-scribe update              install the newest version from PyPI
```

</details>

<details>
<summary><b>Licences</b></summary>

The code is MIT licensed. The models are downloaded at run time and keep their own licences: the Amharic
speech model, [snapwre/hohe-asr-amharic](https://huggingface.co/snapwre/hohe-asr-amharic), is CC-BY-4.0; the
NLLB translation models are CC-BY-NC-4.0, which allows **non-commercial use only**.
`qal-scribe about` lists them all. The ቃል in the logo is drawn from Noto Sans Ethiopic (SIL Open Font
License 1.1).

</details>

<details>
<summary><b>Development</b></summary>

```sh
uv sync
uv run ruff check
uv run pytest                  # no model weights are loaded
uv run python scripts/gpu_lock.py -- uv run pytest -m model     # tests with the real models
uv run python scripts/screenshots.py                            # redraw docs/assets/screens and demo.svg
uv run python eval/fetch_fleurs.py && \
  uv run python scripts/gpu_lock.py -- uv run python eval/evaluate.py eval/data/fleurs-am-test --compare
```

</details>
