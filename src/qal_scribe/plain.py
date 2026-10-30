"""`qal-scribe --plain`: line-based output for logs, scripts and terminals without TUI support."""
import os
import sys
import threading

from qal_scribe import events as ev
from qal_scribe.config import Config
from qal_scribe.controller import Controller


def format_event(e: ev.Event, translation: bool = True) -> str | None:
    match e:
        case ev.Loading(message):
            return f"… {message}"
        case ev.Ready(device, model, _, lm, hotkeys):
            keys = ", ".join(f"hold {k} to {m}" for k, m in hotkeys.items()) or (
                "run `qal-scribe toggle` (bound to a key) to start and stop"
                + (", `qal-scribe toggle --translate` for English" if translation else ""))
            return f"ready on {device} ({model}{', LM on' if lm else ''}): {keys}. Ctrl+C to quit"
        case ev.RecordingStarted(mode):  # with sounds off or `toggle`, the only sign the press was heard
            return f"● recording ({mode})"
        case ev.RecordingStopped(mode, seconds):
            return f"… {seconds:.1f}s recorded, {'translating' if mode == 'translate' else 'transcribing'}"
        case ev.Transcribed(mode, text, _, audio_s, compute_s, pasted):
            tail = "" if pasted or not text else " (not pasted)"
            return f"[{mode} {audio_s:.1f}s audio, {compute_s:.2f}s] {text or '(nothing recognised)'}{tail}"
        case ev.Notice(message, level):
            return f"{level}: {message}"
    return None


def run(cfg: Config) -> None:
    def emit(e: ev.Event) -> None:
        line = format_event(e, cfg.translation)
        if line:
            print(line, flush=True)

    controller = None
    try:
        controller = Controller(cfg, emit)
        controller.start()
        threading.Event().wait()
    except KeyboardInterrupt:
        print("bye", flush=True)
    except (RuntimeError, ValueError) as e:  # already running, unknown mic: say it in one line
        sys.exit(f"qal-scribe: {e}")
    finally:
        if controller and controller.stop() is False:  # a model call (or its download) is still running
            sys.stdout.flush()
            os._exit(0)
