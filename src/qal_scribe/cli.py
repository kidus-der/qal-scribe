"""Command line entry point:
`qal-scribe [--plain] | doctor | models [download [LANG]] | config [--init] | transcribe FILE | about | toggle`.

Errors a user can fix (bad config, no network for the model, unsupported platform) end with one line on stderr
and exit code 1, never a traceback.
"""
import argparse
import logging
import shlex
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

from qal_scribe import __version__
from qal_scribe.config import Config, default_translate_key
from qal_scribe.hotkey import key_label

EPILOG = (f"Run with no command to start dictating: hold {key_label(Config.dictate_key)} to dictate Amharic, "
          f"{key_label(default_translate_key())} for English text. The first run downloads the speech model (about 3 GB).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qal-scribe", description="On-device push-to-talk dictation for Amharic.",
                                epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"qal-scribe {__version__}")
    p.add_argument("--plain", action="store_true", help="line output instead of the TUI")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--no-translate", action="store_true",
                   help="dictate only: no translate key, and the translation model is neither fetched nor loaded")
    sub = p.add_subparsers(dest="command")

    def command(parent, name: str, text: str) -> argparse.ArgumentParser:
        return parent.add_parser(name, help=text, description=text)

    command(sub, "doctor", "check permissions, microphone, device and models")
    m = command(sub, "models", "list available languages and models")
    d = command(m.add_subparsers(dest="models_command"), "download", "fetch model weights now (then it works offline)")
    d.add_argument("language", nargs="?", help="language code (default: the one in your config)")
    d.add_argument("--translation", action="store_true", help="also fetch the Amharic -> English model\n"
                   "(2.5 GB for the default, see `translation_model`)")
    c = command(sub, "config", "show the config file path")
    c.add_argument("--init", action="store_true", help="write a fully commented default config (never overwrites)")
    t = command(sub, "transcribe", "transcribe an audio file and print the text")
    t.add_argument("file", help="wav, flac, ogg or mp3 (m4a: convert it first); any sample rate, mono or stereo")
    t.add_argument("--translate", action="store_true", help="translate to English")
    command(sub, "about", "licences and model attribution")
    tg = command(sub, "toggle", "start/stop recording in the running instance (for Wayland key bindings)")
    tg.add_argument("--translate", action="store_true", help="translate to English")
    return p


def fail(message: str) -> NoReturn:
    sys.exit(f"qal-scribe: {message}")


@contextmanager
def interruptible():
    """Ctrl+C stops a download at once. With Python's handler it would not: the hub joins its download threads
    before the KeyboardInterrupt surfaces, so the whole 3 GB would finish first. Main thread only."""
    old = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


@contextmanager
def friendly_download_errors(required: bool = True):
    """Turn "no network / model not available" into a one-line message with the way out: exit, or with
    required=False a warning on stderr, and the caller goes on without the files."""
    from huggingface_hub.errors import (
        HfHubHTTPError,
        HTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    # Offline, the hub retries for 23 s, logging each retry; the one-line failure below says all of it.
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    try:
        yield
    except (HfHubHTTPError, HTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled) as e:
        reason = str(e).splitlines()[0].removeprefix("Got: ") if str(e) else type(e).__name__
        if not required:
            print(f"qal-scribe: could not get the translation model; dictation works, translation will not yet:\n"
                  f"  {reason}\nConnect to the internet and run `qal-scribe models download --translation`.",
                  file=sys.stderr)
            return
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
        from qal_scribe.translate import TRANSLATION_MODELS

        def state(repo: str, revision: str, *files: str) -> str:
            return "cached" if is_cached(repo, revision, *(files or WEIGHT_FILES)) else "not downloaded"

        rows = [("code", "language", "model", "licence", "state")]
        rows += [(s.language, s.language_name, s.repo, s.license + (" (experimental)" if s.experimental else ""),
                  state(s.repo, s.revision)) for s in models.MODELS.values()]
        rows += [("", f"translation {key}", t.repo, f"{t.license} (Amharic -> English)",
                  state(t.repo, t.revision, t.weights)) for key, t in TRANSLATION_MODELS.items()]
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        for row in rows:
            print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)), row[4], sep="  ")
        print("\ndownload one with: qal-scribe models download LANG [--translation]")
        return
    cfg = load_config(config_path(args.config), explicit=bool(args.config))
    spec = spec_for(args.language or cfg.language)
    with friendly_download_errors(), interruptible():
        print(f"Fetching {spec.language_name} model {spec.repo} ({spec.license})...", flush=True)
        print(f"-> {models.download(spec, with_lm=cfg.lm)}")
        if args.translation:
            from qal_scribe import translate

            t = translation_spec(cfg)
            print(f"Fetching translation model {t.repo} ({t.license})...", flush=True)
            print(f"-> {translate.download(cfg.translation_model)}")


def translation_spec(cfg: Config):
    from qal_scribe.translate import TRANSLATION_MODELS

    if cfg.translation_model not in TRANSLATION_MODELS:
        fail(f"bad config: translation_model {cfg.translation_model!r}: use one of {', '.join(TRANSLATION_MODELS)}")
    return TRANSLATION_MODELS[cfg.translation_model]


def cmd_about() -> None:
    from importlib.resources import files

    print(f"qal-scribe {__version__}, MIT licence\n")
    print(files("qal_scribe").joinpath("NOTICE").read_text(encoding="utf-8"))


def resample(audio, rate: int, to: int):
    """Band-limited (FFT) resampling of a whole file: phones and QuickTime record at 44.1 or 48 kHz."""
    import numpy as np

    # shortcut: the whole file's spectrum in memory (~0.7 GB per hour at 48 kHz); chunk it if hour-long files matter

    n = round(len(audio) * to / rate)
    return (np.fft.irfft(np.fft.rfft(audio), n) * (n / len(audio))).astype("float32")


def cmd_transcribe(cfg: Config, file: str, translate: bool) -> None:
    if not Path(file).is_file():
        fail(f"no such file: {file}")
    if translate and not cfg.translation:
        fail("translation is off (translation = false in the config, or --no-translate); remove --translate")
    import soundfile as sf

    from qal_scribe import postprocess
    from qal_scribe.config import load_words
    from qal_scribe.engine import SAMPLE_RATE, Engine

    check_config(cfg)
    words = load_words(cfg)  # before the model load: a bad words.txt must not cost a 3 GB load first
    try:
        audio, sr = sf.read(file, dtype="float32")
    except sf.SoundFileError as e:  # libsndfile reads no m4a/AAC, which is what Voice Memos and QuickTime save
        quoted = shlex.quote(file)
        convert = (f"afconvert -f WAVE -d LEI16@16000 -c 1 {quoted} out.wav" if sys.platform == "darwin"
                   else f"ffmpeg -i {quoted} -ar 16000 -ac 1 out.wav")
        fail(f"cannot read {file}: {e}\nConvert it with: {convert}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo -> mono
    if sr != SAMPLE_RATE and len(audio):
        audio = resample(audio, sr, SAMPLE_RATE)
    fetch_missing_models(cfg, translation=translate)
    quiet_progress_bars()
    with friendly_download_errors():
        engine = Engine(spec_for(cfg.language), device=cfg.device, use_lm=cfg.lm)
        engine.load()
        result = engine.transcribe(audio, sentences=translate)
        text = postprocess.process(result.text, cfg, words)
        if translate:
            from qal_scribe.translate import NLLB_CODES, Translator

            if cfg.language not in NLLB_CODES:
                fail(f"no translation for {spec_for(cfg.language).language_name}; it knows {', '.join(NLLB_CODES)}")
            tr = Translator(engine.device, model=cfg.translation_model)
            tr.load()
            text = tr.translate([postprocess.process(s, cfg, words) for s in result.sentences], cfg.language)
    print(text)


def fetch_missing_models(cfg: Config, translation: bool = False, translation_required: bool = True) -> None:
    """Download the weights before the UI starts: a failure inside the TUI's worker thread cannot reach
    friendly_download_errors(), and the hub's progress bars only look right in a plain terminal.
    Says so on stderr, which keeps `transcribe`'s stdout to the text alone. translation_required=False: a
    failed translation download is a warning (dictation still works offline)."""
    from qal_scribe import models, translate
    from qal_scribe.doctor import WEIGHT_FILES, is_cached

    spec = spec_for(cfg.language)
    has_weights = is_cached(spec.repo, spec.revision, *WEIGHT_FILES)
    has_lm = not (cfg.lm and spec.lm_file) or is_cached(spec.repo, spec.revision, spec.lm_file)
    if not (has_weights and has_lm):
        print(f"First run: fetching {spec.language_name} model {spec.repo} ({spec.license}), about 3 GB "
              "(Ctrl+C cancels)...", file=sys.stderr, flush=True)
        with friendly_download_errors(), interruptible():
            models.download(spec, with_lm=cfg.lm)
    if not translation:
        return
    t = translation_spec(cfg)
    if not is_cached(t.repo, t.revision, t.weights):
        print(f"First run: fetching the translation model {t.repo} ({t.license}), about {t.size_gb:g} GB "
              "(Ctrl+C cancels)...", file=sys.stderr, flush=True)
        with friendly_download_errors(required=translation_required), interruptible():
            translate.download(cfg.translation_model)


def quiet_progress_bars() -> None:
    """Once the files are downloaded, the hub's "Fetching N files" and transformers' "Loading weights" bars are
    noise, and garbage in a log. Call before transformers is imported: it copies this setting then."""
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()


def check_config(cfg: Config) -> None:
    from qal_scribe.doctor import config_problem

    if problem := config_problem(cfg):
        fail(f"bad config: {problem}\nSee `qal-scribe config` for its location and `qal-scribe doctor` to check it.")


def cmd_run(cfg: Config, plain: bool) -> None:
    if not sys.platform.startswith(("darwin", "linux")):
        fail(f"{sys.platform} is not supported (macOS and Linux only)")
    check_config(cfg)
    # The translation model too: fetched inside the TUI it would be GBs behind a spinner, on a key press.
    fetch_missing_models(cfg, translation=cfg.translation and bool(cfg.translate_key), translation_required=False)
    if plain:
        from qal_scribe.plain import run

        quiet_progress_bars()
    else:
        from qal_scribe.app import run
    with friendly_download_errors():
        run(cfg)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = config_path(args.config)
    # transformers' wav2vec2-bert module imports torch.distributed.elastic, which warns on macOS about a feature
    # qal-scribe never uses.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
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

                if args.config and not path.exists():  # the same check the other commands make
                    fail(f"no config file at {path}")
                sys.exit(run(path))
            case "toggle":
                from qal_scribe.ipc import send_toggle

                sys.exit(send_toggle("translate" if args.translate else "dictate"))
            case "transcribe":
                cfg = load_config(path, bool(args.config))
                cfg.translation = cfg.translation and not args.no_translate
                cmd_transcribe(cfg, args.file, args.translate)
            case _:
                cfg = load_config(path, bool(args.config))
                if args.no_translate:
                    cfg.translation = False
                cmd_run(cfg, args.plain)
    except KeyboardInterrupt:
        sys.exit(130)
