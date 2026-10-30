"""Global push-to-talk keys (macOS/Windows/X11 via pynput). Linux Wayland uses `qal-scribe toggle` instead.

Contract: PushToTalk({"alt_r": "dictate", "cmd_r": "translate"}, on_start(mode), on_stop(mode)).start()
Only one key records at a time; on_start/on_stop run on pynput's thread and must return quickly. `paused = True`
makes the keys do nothing (the settings screen listens for a new key meanwhile).
`reason` is a user-facing sentence when global keys cannot work here (Wayland, no X display, macOS Input
Monitoring off); the UI shows it. `listening` is False when no key will ever be heard (Wayland, no pynput, a
listener that failed to start): `qal-scribe toggle` is then the only way to record.
pynput is only imported when it can work, so this module imports anywhere.
"""
import os
import sys
from collections.abc import Callable

from qal_scribe.events import Mode

# What the key is called on this platform's keyboard: Option/Command on a Mac, Alt/Super elsewhere.
KEY_NAMES = {"ctrl_r": "Right Control", "ctrl_l": "Left Control", "shift_r": "Right Shift", "shift_l": "Left Shift"}
PLATFORM_KEY_NAMES = {
    True: {"alt_r": "Right Option", "alt_l": "Left Option", "cmd_r": "Right Command", "cmd_l": "Left Command"},
    False: {"alt_r": "Right Alt", "alt_l": "Left Alt", "cmd_r": "Right Super", "cmd_l": "Left Super"},
}
WAYLAND_REASON = ("Wayland does not allow global hotkeys: bind `qal-scribe toggle` (and `qal-scribe toggle "
                  "--translate`) to keys in your desktop's keyboard shortcut settings; press once to start "
                  "recording and again to stop")
INPUT_MONITORING_REASON = ("hotkeys need Input Monitoring: add your terminal app under System Settings > Privacy "
                           "& Security > Input Monitoring, then quit and reopen it")


def key_label(name: str) -> str:
    names = KEY_NAMES | PLATFORM_KEY_NAMES[sys.platform == "darwin"]
    return names.get(name) or (name.upper() if name.startswith("f") and name[1:].isdigit() else name)


def is_wayland() -> bool:
    # A compositor started from a console (sway) has XDG_SESSION_TYPE=tty, and DISPLAY too when XWayland runs.
    env = os.environ
    return env.get("XDG_SESSION_TYPE") == "wayland" or bool(env.get("WAYLAND_DISPLAY"))


def input_monitoring_denied() -> bool:
    if sys.platform != "darwin":
        return False
    import Quartz

    return not Quartz.CGPreflightListenEventAccess()


class PushToTalk:
    def __init__(self, keys: dict[str, Mode], on_start: Callable[[Mode], None], on_stop: Callable[[Mode], None]):
        self._on_start, self._on_stop = on_start, on_stop
        self._held = None
        self._keys: dict = {}
        self._listener = None
        self.paused = False
        self.reason: str | None = None  # set when global keys can't work here (e.g. Wayland); UI shows it
        if is_wayland():
            self.reason = WAYLAND_REASON
            return
        if input_monitoring_denied():
            self.reason = INPUT_MONITORING_REASON  # still listen: the first attempt makes macOS show its prompt
        try:
            from pynput import keyboard
        except Exception as e:  # noqa: BLE001 - pynput raises ImportError, or Xlib errors, when there is no X display
            self.reason = f"global hotkeys are unavailable ({type(e).__name__}: {e}); use `qal-scribe toggle`"
            return
        self._keys = {getattr(keyboard.Key, name): mode for name, mode in keys.items()}
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release, daemon=True)

    def _press(self, key) -> None:
        if key in self._keys and self._held is None and not self.paused:
            self._held = key
            self._on_start(self._keys[key])

    def _release(self, key) -> None:
        if key == self._held:
            self._held = None
            self._on_stop(self._keys[key])

    @property
    def listening(self) -> bool:
        return self._listener is not None

    def start(self) -> None:
        if self._listener is None:
            return
        try:
            self._listener.start()
        except Exception as e:  # noqa: BLE001 - e.g. the X server refuses the connection
            self.reason = f"global hotkeys failed to start ({type(e).__name__}: {e}); use `qal-scribe toggle`"
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
