"""`qal-scribe doctor`: one table row per check, exit code 1 if anything blocks dictation (workstream `packaging`).

Contract: run(path) -> int prints the table for the config file at `path` and returns 0, or 1 when a check failed.
config_problem(cfg) -> str | None says why a loaded config cannot start dictation. Checks never record audio,
download anything or write files. Every failing Check carries the exact command in `fix` (and, on macOS, the
Settings pane in `url`); linux_install(*packages) is that command for this distribution.
"""
import importlib.util
import math
import os
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
from qal_scribe.hotkey import is_wayland
from qal_scribe.translate import TRANSLATION_MODELS

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
MIN_FREE_GB = 6  # all models together are about 5.5 GB
STATUS = {"ok": "[green]ok[/]", "warn": "[yellow]warn[/]", "fail": "[bold red]FAIL[/]"}
PREFS = "x-apple.systempreferences:com.apple.preference.security?Privacy_"
PANES = {name: PREFS + name for name in ("Microphone", "Accessibility")} | {"Input Monitoring": PREFS + "ListenEvent"}
TERMINALS = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm2", "ghostty": "Ghostty", "WezTerm": "WezTerm",
             "vscode": "VS Code"}
# package names per distribution family; a name that is the same everywhere is not listed
PACKAGES = {
    "portaudio": {"apt": "libportaudio2", "dnf": "portaudio", "pacman": "portaudio"},
    "build": {"apt": "build-essential python3-dev", "dnf": "gcc python3-devel kernel-headers",
              "pacman": "base-devel", "other": "a C compiler and the Python headers"},
}
INSTALL = {"apt": "sudo apt install", "dnf": "sudo dnf install", "pacman": "sudo pacman -S"}
FAMILIES = {"debian": "apt", "ubuntu": "apt", "fedora": "dnf", "rhel": "dnf", "centos": "dnf", "arch": "pacman"}


@dataclass(frozen=True)
class Check:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str
    fix: str = ""
    url: str = ""  # macOS: the System Settings pane that fixes it


def terminal_name() -> str:
    return TERMINALS.get(os.environ.get("TERM_PROGRAM", ""), "your terminal app")


def grant_fix(pane: str) -> str:
    terminal = terminal_name()
    return f'open "{PANES[pane]}", turn on {terminal}, then quit and reopen {terminal}'


def linux_install(*packages: str) -> str:
    """The command that installs `packages` (names from PACKAGES, or a package that is called the same everywhere)."""
    try:
        release = platform.freedesktop_os_release()
        ids = [release.get("ID", ""), *release.get("ID_LIKE", "").split()]
    except OSError:
        ids = []
    family = next((FAMILIES[i] for i in ids if i in FAMILIES), None)
    names = [PACKAGES.get(p, {}).get(family or "other", p) for p in packages]
    return f"{INSTALL[family]} {' '.join(names)}" if family else f"install: {', '.join(names)}"


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
    if cfg.translation_model not in TRANSLATION_MODELS:
        return f"translation_model {cfg.translation_model!r}: use one of {', '.join(TRANSLATION_MODELS)}"
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
            return f"unknown key {key!r}: use a pynput key name such as alt_r, cmd_r, ctrl_r or f13"
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
        fix = f"fix the line named above in {path}" if path.exists() else "`qal-scribe config --init` writes a fresh one"
        return Check("Config", "fail", problem, fix), Config()
    return Check("Config", "ok", str(path) if path.exists() else "no file, using defaults"), cfg


def check_device(cfg: Config) -> Check:
    """The device dictation will use; an unavailable `device` setting is a config problem (check_config)."""
    import torch

    from qal_scribe.engine import pick_device

    device = pick_device(cfg.device)
    detail = f"{device} (torch {torch.__version__})"
    if device == "cpu" and not cfg.device:
        return Check("Compute device", "warn", detail, "no GPU found: it works, but a 10 s clip takes seconds. To save memory turn "
                     "translation off: `qal-scribe`, then press s")
    return Check("Compute device", "ok", detail)


def check_microphone(mic: str | None = None) -> Check:
    """The input `mic` (config) selects, as Recorder would resolve it; lists the inputs when it matches none."""
    try:
        import sounddevice as sd
    except OSError:  # PortAudio shared library missing
        fix = ("uv tool install --python 3.12 --reinstall qal-scribe" if sys.platform == "darwin"
               else linux_install("portaudio"))  # the macOS wheel bundles PortAudio
        return Check("Microphone", "fail", "PortAudio library not found", fix)
    except Exception as e:  # noqa: BLE001 - importing starts PortAudio, which raises PortAudioError with no audio server
        fix = ("systemctl --user start pipewire pipewire-pulse" if sys.platform.startswith("linux")
               else "check the audio drivers and that an input device is enabled")
        return Check("Microphone", "fail", f"audio system unavailable: {e}", fix)
    device = int(mic) if mic and mic.isdigit() else mic
    try:
        return Check("Microphone", "ok", sd.query_devices(device, kind="input")["name"])
    except (sd.PortAudioError, ValueError):
        if not mic:
            return Check("Microphone", "fail", "no input device",
                         "connect a microphone and make it the default input, then `qal-scribe`, press s, Microphone")
    inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    return Check("Microphone", "fail", f"no input matches mic = {mic!r}",
                 f"`qal-scribe`, press s, Microphone; or set mic to one of: {', '.join(inputs)}; or remove it" if inputs
                 else "connect a microphone")


def check_paste() -> Check:
    from qal_scribe import paste

    ok, reason = paste.can_paste()
    if sys.platform == "darwin":
        name = "Accessibility"
        if not ok:
            return Check(name, "fail", f"{terminal_name()} cannot paste yet", grant_fix(name), PANES[name])
    elif ok:
        name = "Text insertion"
    elif sys.platform.startswith("linux"):
        name = "Text insertion"
        fix = (f"{linux_install('wtype')} (sway, Hyprland), or {linux_install('ydotool', 'wl-clipboard')} and start "
               "ydotoold (GNOME, KDE)" if is_wayland() else linux_install("xdotool", "xclip"))
        return Check(name, "fail", reason, fix)
    else:
        return Check("Text insertion", "fail", "cannot insert text", reason)
    return Check(name, "ok", reason)


def check_hotkeys() -> Check | None:
    """Linux only: macOS reports hotkeys as Input Monitoring."""
    if not sys.platform.startswith("linux"):
        return None
    if is_wayland():
        return Check("Hotkeys", "warn", "Wayland has no global hotkeys",
                     "bind `qal-scribe toggle` and `qal-scribe toggle --translate` in your desktop's keyboard settings")
    try:
        import pynput.keyboard  # noqa: F401 - the import is the test: it fails without an X display
    except Exception as e:  # noqa: BLE001 - pynput raises ImportError, or Xlib errors, when there is no X display
        return Check("Hotkeys", "fail", f"{type(e).__name__}: {e}", "run qal-scribe in an X11 or Wayland session")
    return Check("Hotkeys", "ok", "global keys work (X11)")


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
                     grant_fix("Microphone"), PANES["Microphone"])
    return Check("Microphone permission", "ok", "macOS asks the first time you record" if status == 0 else "granted")


def check_input_monitoring() -> Check | None:
    if sys.platform != "darwin":
        return None
    import Quartz

    if Quartz.CGPreflightListenEventAccess():
        return Check("Input Monitoring", "ok", "permission granted")
    return Check("Input Monitoring", "fail", "the hotkeys cannot hear the keyboard",
                 grant_fix("Input Monitoring"), PANES["Input Monitoring"])


def check_models(cfg: Config) -> list[Check]:
    spec = models.get(cfg.language)
    fetch = "`qal-scribe models download`, or `qal-scribe setup`"
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
    if not cfg.translation:
        return Check("Translation model", "ok", "off (translation = false)")
    if not cfg.translate_key:
        return Check("Translation model", "ok", "off (translate_key is empty)")
    spec = TRANSLATION_MODELS[cfg.translation_model]
    if is_cached(spec.repo, spec.revision, spec.weights):
        return Check("Translation model", "ok", f"{spec.repo} cached")
    return Check("Translation model", "warn", f"{spec.repo} not downloaded (about {spec.size_gb:g} GB)",
                 "`qal-scribe models download --translation`, or `qal-scribe setup`")


def check_disk(cfg: Config | None = None) -> Check:
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(HF_HUB_CACHE)
    while not folder.exists():
        folder = folder.parent
    free = shutil.disk_usage(folder).free / 1e9
    detail = f"{free:.0f} GB free where models are stored"
    need = MIN_FREE_GB
    if cfg and cfg.translation:  # the speech models are about 3.5 GB
        need = max(need, 3.5 + TRANSLATION_MODELS[cfg.translation_model].size_gb)
    if free < need:
        return Check("Disk space", "warn", detail, f"models need up to {need:g} GB; free some space or set HF_HOME")
    return Check("Disk space", "ok", detail)


def run(path: Path) -> int:
    console = Console()
    with console.status("checking…"):  # the first torch import after an install takes ~10 s
        config_check, cfg = check_config(path)
        checks = [check_system(), check_device(cfg), check_microphone(cfg.mic), check_microphone_permission(),
                  check_paste(), check_input_monitoring(), check_hotkeys(), config_check, *check_models(cfg), check_disk(cfg)]
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
