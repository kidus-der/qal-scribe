# Install

## What you need

| | |
|---|---|
| **Computer** | a Mac with Apple Silicon (M1 or later). Intel Macs are not supported: PyTorch no longer builds for them. Linux works and is in beta. |
| **Disk** | about 6 GB free: 3 GB for the speech model, 2.5 GB more for translation (more for the bigger translation models). |
| **uv** | the Python tool installer. You do not need Python itself: uv fetches Python 3.12. |
| **Internet** | only for the first download. After that qal-scribe works offline. |

If you do not have uv yet:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal window so it is on your `PATH` ([other ways to install uv](https://docs.astral.sh/uv/getting-started/installation/)).

## macOS

```sh
uv tool install --python 3.12 qal-scribe
qal-scribe
```

The first start opens the setup screen: turn on the permissions it lists ([what each one is for](permissions.md)),
choose whether you want translation, and press **Download**. When the bars are full, press **Continue**.

`--python 3.12` matters: the language-model decoder, which makes the text a little more accurate, has no builds
for newer Pythons. Without it qal-scribe still works, with slightly less accurate decoding, and
`qal-scribe doctor` warns about it.

## Linux (beta)

1. Install PortAudio and the tools to build `evdev`, a keyboard library with no ready-made build for Linux:

   ```sh
   sudo apt install libportaudio2 build-essential python3-dev      # Debian, Ubuntu
   sudo dnf install portaudio gcc python3-devel kernel-headers      # Fedora
   sudo pacman -S portaudio base-devel                             # Arch
   ```

2. Install qal-scribe. Without an NVIDIA GPU, add `--torch-backend cpu` to skip the multi-GB CUDA build of PyTorch:

   ```sh
   uv tool install --python 3.12 qal-scribe --torch-backend cpu
   ```

3. Run `qal-scribe`. The setup screen shows the install command for anything still missing, such as the tool
   that types the text ([details](permissions.md#linux)).

## Download without starting

```sh
qal-scribe models download                  # the speech model
qal-scribe models download --translation    # the speech model and the translation model
```

Ctrl+C cancels a download; running the command again begins it again.

## Check everything

```sh
qal-scribe doctor
```

It checks the permissions, microphone, compute device, config, models and disk space, and prints the fix
next to anything that is wrong.

## Upgrade

```sh
uv tool upgrade qal-scribe
```

On Linux with the CPU build, upgrade with the install line plus `--upgrade` instead: uv does not remember
`--torch-backend`, and `uv tool upgrade` may fetch the CUDA build.

```sh
uv tool install --python 3.12 qal-scribe --torch-backend cpu --upgrade
```

## Uninstall

```sh
uv tool uninstall qal-scribe
```

That leaves your settings and the downloaded models. To remove them too, delete:

- the settings: `~/.config/qal-scribe` (the folder `qal-scribe config` shows)
- the models: the `models--snapwre--hohe-asr-amharic` and `models--facebook--nllb-*` (or `models--google--madlad400-*`)
  folders in `~/.cache/huggingface/hub` (or under `$HF_HOME`)
- a faster-loading copy of the speech model: `~/Library/Caches/qal-scribe` on macOS, `~/.cache/qal-scribe` on Linux
