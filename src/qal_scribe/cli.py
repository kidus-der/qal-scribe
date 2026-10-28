"""Command line entry point: `qal-scribe [--plain] | doctor | models | transcribe FILE | about | toggle`."""
import argparse
import sys

from qal_scribe import __version__


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="qal-scribe", description="On-device push-to-talk dictation for Amharic.")
    p.add_argument("--version", action="version", version=f"qal-scribe {__version__}")
    p.add_argument("--plain", action="store_true", help="line output instead of the TUI")
    p.add_argument("--config", help="path to config.toml")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check permissions, microphone, device and model")
    sub.add_parser("models", help="list available languages and models")
    t = sub.add_parser("transcribe", help="transcribe a 16 kHz mono wav file and print the text")
    t.add_argument("file")
    t.add_argument("--translate", action="store_true", help="translate to English")
    sub.add_parser("about", help="licences and model attribution")
    sub.add_parser("toggle", help="start/stop recording in the running instance (for Wayland key bindings)")
    args = p.parse_args(argv)

    from pathlib import Path

    from qal_scribe import config

    cfg = config.load(Path(args.config) if args.config else None)

    if args.command == "models":
        from qal_scribe.models import MODELS

        for spec in MODELS.values():
            flag = " (experimental)" if spec.experimental else ""
            print(f"{spec.language}  {spec.language_name:<10} {spec.repo}  {spec.license}{flag}")
    elif args.command == "about":
        from importlib.resources import files

        print(f"qal-scribe {__version__}, MIT licence\n")
        print(files("qal_scribe").joinpath("NOTICE").read_text(encoding="utf-8"))
    elif args.command == "doctor":
        from qal_scribe.doctor import run as doctor

        sys.exit(doctor(cfg))
    elif args.command == "transcribe":
        import soundfile as sf

        from qal_scribe import models, postprocess
        from qal_scribe.engine import SAMPLE_RATE, Engine

        audio, sr = sf.read(args.file, dtype="float32")
        if sr != SAMPLE_RATE or audio.ndim != 1:
            sys.exit(f"need {SAMPLE_RATE} Hz mono audio, got {sr} Hz with {audio.ndim} channel(s)")
        engine = Engine(models.get(cfg.language), device=cfg.device, use_lm=cfg.lm)
        engine.load()
        text = postprocess.process(engine.transcribe(audio).text, cfg)
        if args.translate:
            from qal_scribe.translate import Translator

            tr = Translator(engine.device)
            tr.load()
            text = tr.translate(text)
        print(text)
    elif args.command == "toggle":
        from qal_scribe.ipc import send_toggle

        sys.exit(send_toggle())
    elif args.plain:
        from qal_scribe.plain import run

        run(cfg)
    else:
        from qal_scribe.app import run

        run(cfg)
