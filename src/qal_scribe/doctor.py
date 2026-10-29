"""`qal-scribe doctor`: one table row per check, exit code 1 if anything blocks dictation (workstream `packaging`).

Contract: run(path) -> int prints the table for the config file at `path` and returns 0, or 1 when a check failed.
config_problem(cfg) -> str | None says why a loaded config cannot start dictation. Checks never record audio,
download anything or write files.
"""
import importlib.util
import math
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from qal_scribe import config as config_mod
from qal_scribe import models
from qal_scribe.config import Config

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
MIN_FREE_GB = 6  # all models together are about 5.5 GB
STATUS = {"ok": "[green]ok[/]", "warn": "[yellow]warn[/]", "fail": "[bold red]FAIL[/]"}


@dataclass(frozen=True)
class Check:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str
    fix: str = ""


def is_cached(repo: str, revision: str, *files: str) -> bool:
    """True when any of `files` of the pinned revision is already in the Hugging Face cache (no network)."""
    from huggingface_hub import try_to_load_from_cache

    return any(isinstance(try_to_load_from_cache(repo, f, revision=revision), str) for f in files)


def available_devices() -> list[str]:
    import torch

    gpus = (("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available()))
    return ["cpu", *(d for d, ok in gpus if ok)]


def config_problem(cfg: Config) -> str | None:
    """Why this config cannot start dictation, or None. Imports torch when `device` is set."""
    try:
        models.get(cfg.language)
        config_mod.load_words(cfg)
    except ValueError as e:
        return str(e)
    if not math.isfinite(cfg.restore_delay) or cfg.restore_delay < 0:
        return f"restore_delay is {cfg.restore_delay}: it must be a number of seconds, 0 or more"
    if cfg.device and cfg.device.split(":")[0] not in (available := available_devices()):
        return f"device {cfg.device!r} is not available here: use one of {', '.join(available)}, or remove it"
    if not cfg.dictate_key:
        return "dictate_key is empty: set it to a key name such as alt_r (only translate_key can be turned off)"
    if cfg.translate_key and cfg.translate_key == cfg.dictate_key:
        return f"dictate_key and translate_key are both {cfg.dictate_key!r}"
    try:
        from pynput import keyboard
    except ImportError:
        return None  # no display here (e.g. headless Linux); the platform checks report that
    for key in (cfg.dictate_key, cfg.translate_key):
        if key and key not in keyboard.Key.__members__:
            return f"unknown key {key!r}: use a pynput key name such as alt_r, ctrl_r, cmd_r or f13"
    return None


def check_system() -> Check:
    where = f"{platform.machine()}, Python {platform.python_version()}"
    if sys.platform == "darwin":
        return Check("System", "ok", f"macOS {platform.mac_ver()[0]} {where}")
    if sys.platform.startswith("linux"):
        return Check("System", "warn", f"Linux {where}", "Linux support is in beta; please report problems")
    return Check("System", "fail", f"{sys.platform}, {where}", "qal-scribe supports macOS and Linux (beta)")


def check_config(path: Path) -> tuple[Check, Config]:
    """The config row, plus the config the other checks should use (defaults when the file is unusable)."""
    try:
        cfg = config_mod.load(path)
        problem = config_problem(cfg)
    except (ValueError, OSError) as e:
        cfg, problem = Config(), str(e)
    if problem:  # the other checks cannot run on a config like this (unknown language, ...)
        return Check("Config", "fail", problem, "edit it (`qal-scribe config` shows where it is)"), Config()
    return Check("Config", "ok", str(path) if path.exists() else "no file, using defaults"), cfg


def check_device(cfg: Config) -> Check:
    """The device dictation will use; an unavailable `device` setting is a config problem (check_config)."""
    import torch

    from qal_scribe.engine import pick_device

    device = pick_device(cfg.device)
    detail = f"{device} (torch {torch.__version__})"
    if device == "cpu" and not cfg.device:
        return Check("Compute device", "warn", detail, "no GPU found: it works, but a 10 s clip takes seconds")
    return Check("Compute device", "ok", detail)


def check_microphone(mic: str | None = None) -> Check:
    """The input `mic` (config) selects, as Recorder would resolve it; lists the inputs when it matches none."""
    try:
        import sounddevice as sd
    except OSError:  # PortAudio shared library missing
        return Check("Microphone", "fail", "PortAudio library not found", "Linux: sudo apt install libportaudio2")
    device = int(mic) if mic and mic.isdigit() else mic
    try:
        return Check("Microphone", "ok", sd.query_devices(device, kind="input")["name"])
    except (sd.PortAudioError, ValueError):
        if not mic:
            return Check("Microphone", "fail", "no input device", "connect a microphone and make it the default input")
    inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    return Check("Microphone", "fail", f"no input matches mic = {mic!r}",
                 f"set mic to one of: {', '.join(inputs)}; or remove it" if inputs else "connect a microphone")


def check_paste() -> Check:
    from qal_scribe import paste

    name = "Accessibility" if sys.platform == "darwin" else "Text insertion"
    ok, reason = paste.can_paste()
    return Check(name, "ok", reason) if ok else Check(name, "fail", "cannot insert text", reason)


def microphone_permission() -> int:
    """macOS AVAuthorizationStatus for audio: 0 not asked yet, 1 restricted, 2 denied, 3 allowed. Never prompts."""
    import objc
    from Foundation import NSBundle

    NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework").load()
    return objc.lookUpClass("AVCaptureDevice").authorizationStatusForMediaType_("soun")  # AVMediaTypeAudio


def check_microphone_permission() -> Check | None:
    if sys.platform != "darwin":
        return None
    status = microphone_permission()
    if status in (1, 2):
        return Check("Microphone permission", "fail", "macOS gives the terminal silence instead of your voice",
                     "System Settings > Privacy & Security > Microphone: turn on your terminal app, then quit and "
                     "reopen it")
    return Check("Microphone permission", "ok", "macOS asks the first time you record" if status == 0 else "granted")


def check_input_monitoring() -> Check | None:
    if sys.platform != "darwin":
        return None
    import Quartz

    if Quartz.CGPreflightListenEventAccess():
        return Check("Input Monitoring", "ok", "permission granted")
    return Check("Input Monitoring", "fail", "the hotkeys cannot hear the keyboard",
                 "System Settings > Privacy & Security > Input Monitoring: add your terminal app, "
                 "then quit and reopen it")


def check_models(cfg: Config) -> list[Check]:
    spec = models.get(cfg.language)
    fetch = "`qal-scribe models download`, or let the first run fetch it"
    asr = (Check("Speech model", "ok", f"{spec.repo} cached") if is_cached(spec.repo, spec.revision, *WEIGHT_FILES)
           else Check("Speech model", "warn", f"{spec.repo} not downloaded (about 2.4 GB)", fetch))
    if not cfg.lm or not spec.lm_file:
        lm = Check("Language model", "ok", "off" if not cfg.lm else "none for this language")
    elif importlib.util.find_spec("flashlight") is None:
        lm = Check("Language model", "warn", f"its decoder (flashlight-text) has no build for Python "
                   f"{platform.python_version()}, so decoding is greedy and a little less accurate",
                   "reinstall on Python 3.12: uv tool install --python 3.12 --reinstall qal-scribe")
    elif is_cached(spec.repo, spec.revision, spec.lm_file):
        lm = Check("Language model", "ok", f"{spec.lm_file} cached")
    else:
        lm = Check("Language model", "warn", f"{spec.lm_file} not downloaded (about 0.5 GB)", fetch)
    return [asr, lm, check_translation(cfg)]


def check_translation(cfg: Config) -> Check:
    from qal_scribe.translate import Translator

    if not cfg.translate_key:
        return Check("Translation model", "ok", "off (translate_key is empty)")
    if is_cached(Translator.MODEL, Translator.REVISION, *WEIGHT_FILES):
        return Check("Translation model", "ok", f"{Translator.MODEL} cached")
    return Check("Translation model", "warn", f"{Translator.MODEL} not downloaded (about 2.5 GB)",
                 "`qal-scribe models download --translation`, or let the first translation fetch it")


def check_disk() -> Check:
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(HF_HUB_CACHE)
    while not folder.exists():
        folder = folder.parent
    free = shutil.disk_usage(folder).free / 1e9
    detail = f"{free:.0f} GB free where models are stored"
    if free < MIN_FREE_GB:
        return Check("Disk space", "warn", detail, f"models need up to {MIN_FREE_GB} GB; free some space or set HF_HOME")
    return Check("Disk space", "ok", detail)


def run(path: Path) -> int:
    console = Console()
    with console.status("checking…"):  # the first torch import after an install takes ~10 s
        config_check, cfg = check_config(path)
        checks = [check_system(), check_device(cfg), check_microphone(cfg.mic), check_microphone_permission(),
                  check_paste(), check_input_monitoring(), config_check, *check_models(cfg), check_disk()]
    checks = [c for c in checks if c]
    table = Table(title="qal-scribe doctor", title_justify="left", header_style="bold")
    with_fix = any(c.fix for c in checks)
    for column in ("Check", "Status", "Result", *(("Fix",) if with_fix else ())):
        table.add_column(column, overflow="fold")
    for c in checks:
        table.add_row(c.name, STATUS[c.status], escape(c.detail), *((escape(c.fix),) if with_fix else ()))
    console.print(table)
    failed = [c.name for c in checks if c.status == "fail"]
    warned = sum(c.status == "warn" for c in checks)
    if failed:
        console.print(f"[bold red]Dictation is blocked:[/] fix {', '.join(failed)}.")
    else:
        console.print(f"[green]Ready to dictate.[/]{f' {warned} warning(s) above.' if warned else ''}")
    return 1 if failed else 0
