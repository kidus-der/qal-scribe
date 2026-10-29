"""Linux (BETA): insert text with wtype (Wayland), ydotool (Wayland, GNOME/KDE) or xdotool (X11).

Contract: same as paste/__init__. Typing backends leave the clipboard alone. When nothing can type, the text is
put on the clipboard (wl-copy / xclip / xsel) and paste() returns False so the UI can tell the user to paste it.
`ydotool type` only knows the US layout (no Amharic), so the ydotool backend pastes: it copies the text with
wl-copy, sends Ctrl+V as raw keycodes, then restores the previous text clipboard.
"""
import shutil
import subprocess
import time

from qal_scribe.hotkey import is_wayland

TIMEOUT = 5
PER_CHAR = 0.05  # typing backends: xdotool waits 12 ms per key, plus a keymap remap for every Fidel character
COPIERS = {"wl-copy": ["wl-copy"], "xclip": ["xclip", "-selection", "clipboard"],
           "xsel": ["xsel", "--clipboard", "--input"]}
CTRL_V = ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]  # KEY_LEFTCTRL, KEY_V: layout independent
DESCRIPTIONS = {
    "wtype": "typing with wtype (Wayland, beta)",
    "ydotool": "pasting with wl-copy + ydotool Ctrl+V (Wayland, beta): ydotoold must be running, and terminals "
               "need Ctrl+Shift+V so they will not receive the text",
    "xdotool": "typing with xdotool (X11, beta)",
}


def _have(*tools: str) -> bool:
    return all(shutil.which(t) for t in tools)


def backends() -> list[str]:
    """Insertion methods usable here, best first; paste() tries them in order."""
    if not is_wayland():
        return ["xdotool"] if _have("xdotool") else []
    return [name for name, needs in (("wtype", ["wtype"]), ("ydotool", ["ydotool", "wl-copy"])) if _have(*needs)]


def _copier() -> list[str] | None:
    names = ["wl-copy"] if is_wayland() else ["xclip", "xsel"]
    return next((COPIERS[n] for n in names if _have(n)), None)


def can_paste() -> tuple[bool, str]:
    found = backends()
    if found:
        return True, DESCRIPTIONS[found[0]]
    if is_wayland():
        need = "wtype (sway, Hyprland, ...) or ydotool + wl-clipboard (GNOME, KDE)"
    else:
        need = "xdotool"
    clip = "" if _copier() else "; there is no clipboard tool either (wl-clipboard, xclip or xsel)"
    return False, f"cannot type text: install {need}. Until then the text is copied to the clipboard{clip}"


def _run(argv: list[str], data: bytes | None = None, timeout: float = TIMEOUT) -> bool:
    # DEVNULL, not pipes: xclip and wl-copy fork into the background and would keep captured pipes open forever.
    try:
        done = subprocess.run(argv, check=False, input=data, timeout=timeout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _copy(text: str) -> bool:
    argv = _copier()
    return argv is not None and _run(argv, text.encode())


def _clipboard_text() -> bytes | None:
    try:
        done = subprocess.run(["wl-paste", "--no-newline", "--type", "text"], check=False, capture_output=True,
                              timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 and done.stdout else None


def _ydotool_paste(text: str, restore_delay: float) -> bool:
    saved = _clipboard_text()  # text only: an image or file on the clipboard is left alone (and replaced by ours)
    if not (_copy(text) and _run(CTRL_V)):
        return False
    time.sleep(restore_delay)  # apps read the clipboard a moment after Ctrl+V
    if saved is not None and _clipboard_text() == text.encode():  # don't clobber a copy made in the meantime
        _run(COPIERS["wl-copy"], saved)
    return True


def _insert(backend: str, text: str, restore_delay: float) -> bool:
    # shortcut: a typing backend killed at its timeout has typed part of the text, and the next backend then
    # inserts all of it again; the timeout grows with the text so only a hung tool gets there.
    typing_time = TIMEOUT + PER_CHAR * len(text)
    if backend == "wtype":  # "-" reads stdin: no argv length or leading-dash problems
        return _run(["wtype", "-"], text.encode(), typing_time)
    if backend == "xdotool":
        return _run(["xdotool", "type", "--clearmodifiers", "--", text], timeout=typing_time)
    return _ydotool_paste(text, restore_delay)


def paste(text: str, restore_delay: float = 0.5) -> bool:
    if any(_insert(b, text, restore_delay) for b in backends()):
        return True
    _copy(text)
    return False
