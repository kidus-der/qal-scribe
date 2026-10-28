"""`qal-scribe --plain`: line-based output for logs, scripts and terminals without TUI support."""
import threading

from qal_scribe import events as ev
from qal_scribe.config import Config
from qal_scribe.controller import Controller


def format_event(e: ev.Event) -> str | None:
    match e:
        case ev.Loading(message):
            return f"… {message}"
        case ev.Ready(device, model, _, lm, hotkeys):
            keys = ", ".join(f"hold {k} to {m}" for k, m in hotkeys.items())
            return f"ready on {device} ({model}{', LM on' if lm else ''}): {keys}. Ctrl+C to quit"
        case ev.Transcribed(mode, text, _, audio_s, compute_s, pasted):
            tail = "" if pasted else " (not pasted)"
            return f"[{mode} {audio_s:.1f}s audio, {compute_s:.2f}s] {text}{tail}"
        case ev.Notice(message, level):
            return f"{level}: {message}"
    return None


def run(cfg: Config) -> None:
    def emit(e: ev.Event) -> None:
        line = format_event(e)
        if line:
            print(line, flush=True)

    Controller(cfg, emit).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("bye")
