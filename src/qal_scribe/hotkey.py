"""Global push-to-talk keys (macOS/Windows/X11 via pynput). Linux Wayland uses `qal-scribe toggle` instead.

Contract: PushToTalk({"alt_r": "dictate", "ctrl_r": "translate"}, on_start(mode), on_stop(mode)).start()
Only one key records at a time; on_start/on_stop run on pynput's thread and must return quickly.
"""
from collections.abc import Callable

from qal_scribe.events import Mode

KEY_NAMES = {"alt_r": "Right Option", "ctrl_r": "Right Control", "cmd_r": "Right Command", "alt_l": "Left Option"}


def key_label(name: str) -> str:
    return KEY_NAMES.get(name, name)


class PushToTalk:
    def __init__(self, keys: dict[str, Mode], on_start: Callable[[Mode], None], on_stop: Callable[[Mode], None]):
        from pynput import keyboard

        self._keys = {getattr(keyboard.Key, name): mode for name, mode in keys.items()}
        self._on_start, self._on_stop = on_start, on_stop
        self._held = None
        self.reason: str | None = None  # set when global keys can't work here (e.g. Wayland); UI shows it
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release, daemon=True)

    def _press(self, key) -> None:
        if key in self._keys and self._held is None:
            self._held = key
            self._on_start(self._keys[key])

    def _release(self, key) -> None:
        if key == self._held:
            self._held = None
            self._on_stop(self._keys[key])

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
