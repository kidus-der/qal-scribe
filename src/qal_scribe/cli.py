"""Command line entry point:
`qal-scribe [--plain] | doctor | models [download [LANG]] | config [--init] | transcribe FILE | about | toggle`.

Errors a user can fix (bad config, no network for the model, unsupported platform) end with one line on stderr
and exit code 1, never a traceback.
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

from qal_scribe import __version__
from qal_scribe.config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qal-scribe", description="On-device push-to-talk dictation for Amharic.")
    p.add_argument("--version", action="version", version=f"qal-scribe {__version__}")
    p.add_argument("--plain", action="store_true", help="line output instead of the TUI")
    p.add_argument("--config", help="path to config.toml")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check permissions, microphone, device and models")
    m = sub.add_parser("models", help="list available languages and models")
    d = m.add_subparsers(dest="models_command").add_parser("download", help="fetch model weights now (then it works offline)")
    d.add_argument("language", nargs="?", help="language code (default: the one in your config)")
    d.add_argument("--translation", action="store_true", help="also fetch the Amharic -> English model (2.5 GB)")
    c = sub.add_parser("config", help="show the config file path")
    c.add_argument("--init", action="store_true", help="write a fully commented default config (never overwrites)")
    t = sub.add_parser("transcribe", help="transcribe a 16 kHz mono wav file and print the text")
    t.add_argument("file")
    t.add_argument("--translate", action="store_true", help="translate to English")
    sub.add_parser("about", help="licences and model attribution")
    sub.add_parser("toggle", help="start/stop recording in the running instance (for Wayland key bindings)")
    return p


def fail(message: str) -> NoReturn:
    sys.exit(f"qal-scribe: {message}")


@contextmanager
def friendly_download_errors():
    """Turn "no network / model not available" into a one-line message with the way out."""
    from huggingface_hub.errors import (
        HfHubHTTPError,
        HTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    try:
        yield
    except (HfHubHTTPError, HTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled) as e:
        reason = str(e).splitlines()[0] if str(e) else type(e).__name__
        fail(f"could not get the model files from Hugging Face:\n  {reason}\nConnect to the internet and run "
             "`qal-scribe models download`; once downloaded, qal-scribe works offline.")


def config_path(arg: str | None) -> Path:
    from qal_scribe import config

    return Path(arg).expanduser() if arg else config.config_dir() / "config.toml"


def load_config(path: Path, explicit: bool) -> Config:
    from qal_scribe import config

    if explicit and not path.exists():
        fail(f"no config file at {path}")
    try:
        return config.load(path)
    except (ValueError, OSError) as e:  # TOMLDecodeError is a ValueError
        what = str(e) if str(path) in str(e) else f"{path}: {e}"
        fail(f"bad config, {what}\nFix it, or see `qal-scribe config --init` for a commented example.")


def spec_for(language: str):
    from qal_scribe import models

    try:
        return models.get(language)
    except ValueError as e:
        fail(str(e))


def cmd_config(path: Path, init: bool) -> None:
    if not init:
        print(path)
        if not path.exists():
            print("(no file yet: `qal-scribe config --init` writes a commented one)", file=sys.stderr)
        return
    from importlib.resources import files

    template = files("qal_scribe").joinpath("default_config.toml").read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:  # "x": refuse to overwrite, atomically
            f.write(template)
    except FileExistsError:
        fail(f"{path} already exists; not overwriting it (delete it first to start over)")
    print(f"wrote {path}")


def cmd_models(args: argparse.Namespace) -> None:
    from qal_scribe import models
    from qal_scribe.doctor import WEIGHT_FILES, is_cached

    if args.models_command != "download":
        for spec in models.MODELS.values():
            flag = " (experimental)" if spec.experimental else ""
            state = "cached" if is_cached(spec.repo, spec.revision, *WEIGHT_FILES) else "not downloaded"
            print(f"{spec.language}  {spec.language_name:<10} {spec.repo}  {spec.license}{flag}  [{state}]")
        return
    cfg = load_config(config_path(args.config), explicit=bool(args.config))
    spec = spec_for(args.language or cfg.language)
    with friendly_download_errors():
        print(f"Fetching {spec.language_name} model {spec.repo} ({spec.license})...", flush=True)
        print(f"-> {models.download(spec, with_lm=cfg.lm)}")
        if args.translation:
            from huggingface_hub import snapshot_download

            from qal_scribe.translate import Translator

            print(f"Fetching translation model {Translator.MODEL} (CC-BY-NC-4.0, non-commercial)...", flush=True)
            patterns = ["*.json", "*.bin", "*.model"]  # same files Translator.load() asks for
            print(f"-> {snapshot_download(Translator.MODEL, revision=Translator.REVISION, allow_patterns=patterns)}")


def cmd_about() -> None:
    from importlib.resources import files

    print(f"qal-scribe {__version__}, MIT licence\n")
    print(files("qal_scribe").joinpath("NOTICE").read_text(encoding="utf-8"))


def cmd_transcribe(cfg: Config, file: str, translate: bool) -> None:
    import soundfile as sf

    from qal_scribe import postprocess
    from qal_scribe.engine import SAMPLE_RATE, Engine

    try:
        audio, sr = sf.read(file, dtype="float32")
    except sf.SoundFileError as e:
        fail(f"cannot read {file}: {e}")
    if sr != SAMPLE_RATE or audio.ndim != 1:
        fail(f"need {SAMPLE_RATE} Hz mono audio, got {sr} Hz with {audio.ndim} channel(s)")
    with friendly_download_errors():
        engine = Engine(spec_for(cfg.language), device=cfg.device, use_lm=cfg.lm)
        engine.load()
        text = postprocess.process(engine.transcribe(audio).text, cfg)
        if translate:
            from qal_scribe.translate import Translator

            tr = Translator(engine.device)
            tr.load()
            text = tr.translate(text)
    print(text)


def fetch_missing_models(cfg: Config) -> None:
    """Download the weights before the UI starts: a failure inside the TUI's worker thread cannot reach
    friendly_download_errors(), and the hub's progress bars only look right in a plain terminal."""
    from qal_scribe import models
    from qal_scribe.doctor import WEIGHT_FILES, is_cached

    spec = spec_for(cfg.language)
    has_weights = is_cached(spec.repo, spec.revision, *WEIGHT_FILES)
    has_lm = not (cfg.lm and spec.lm_file) or is_cached(spec.repo, spec.revision, spec.lm_file)
    if has_weights and has_lm:
        return
    with friendly_download_errors():
        print(f"First run: fetching {spec.language_name} model {spec.repo} ({spec.license}), about 3 GB...", flush=True)
        models.download(spec, with_lm=cfg.lm)


def cmd_run(cfg: Config, plain: bool) -> None:
    if not sys.platform.startswith(("darwin", "linux")):
        fail(f"{sys.platform} is not supported (macOS and Linux only)")
    from qal_scribe.doctor import config_problem

    if problem := config_problem(cfg):
        fail(f"bad config: {problem}\nSee `qal-scribe config` for its location and `qal-scribe doctor` to check it.")
    fetch_missing_models(cfg)
    if plain:
        from qal_scribe.plain import run
    else:
        from qal_scribe.app import run
    with friendly_download_errors():
        run(cfg)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = config_path(args.config)
    try:
        match args.command:
            case "config":
                cmd_config(path, args.init)
            case "models":
                cmd_models(args)
            case "about":
                cmd_about()
            case "doctor":
                from qal_scribe.doctor import run

                sys.exit(run(path))
            case "toggle":
                from qal_scribe.ipc import send_toggle

                sys.exit(send_toggle())
            case "transcribe":
                cmd_transcribe(load_config(path, bool(args.config)), args.file, args.translate)
            case _:
                cmd_run(load_config(path, bool(args.config)), args.plain)
    except KeyboardInterrupt:
        sys.exit(130)
