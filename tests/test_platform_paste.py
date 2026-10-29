import subprocess

import pytest

from qal_scribe.paste import linux

TEXT = "ሰላም ዓለም "


class Box:
    """Fake machine: which tools are installed, what `subprocess.run` was asked, which tools fail."""

    def __init__(self, monkeypatch, tools, wayland, failing=(), clipboard=b"old"):
        self.tools, self.failing, self.clipboard = set(tools), set(failing), clipboard
        self.calls = []
        for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland" if wayland else "x11")
        monkeypatch.setattr(linux.shutil, "which", lambda t: f"/usr/bin/{t}" if t in self.tools else None)
        monkeypatch.setattr(linux.subprocess, "run", self.run)
        monkeypatch.setattr(linux.time, "sleep", lambda s: None)

    def run(self, argv, input=None, **kw):
        self.calls.append((argv[0], argv[1:], input))
        if argv[0] in self.failing:
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        if argv[0] == "wl-copy":
            self.clipboard = input
        out = self.clipboard if argv[0] == "wl-paste" else b""
        return subprocess.CompletedProcess(argv, 0, out, b"")

    def tools_called(self):
        return [c[0] for c in self.calls]


def test_x11_types_with_xdotool(monkeypatch):
    box = Box(monkeypatch, {"xdotool", "xclip"}, wayland=False)
    assert linux.can_paste() == (True, "typing with xdotool (X11, beta)")
    assert linux.paste(TEXT) is True
    assert box.calls == [("xdotool", ["type", "--clearmodifiers", "--", TEXT], None)]  # clipboard untouched


def test_typing_timeout_grows_with_the_text(monkeypatch):
    timeouts = []
    Box(monkeypatch, {"xdotool"}, wayland=False)
    monkeypatch.setattr(linux.subprocess, "run", lambda argv, **kw: timeouts.append(kw["timeout"]) or
                        subprocess.CompletedProcess(argv, 0, b"", b""))
    linux.paste("ሀ" * 10)
    linux.paste("ሀ" * 1000)  # about a minute of dictation: xdotool needs far longer than the 5 s default
    assert timeouts[0] < 6 and timeouts[1] >= 1000 * 0.012 * 2


def test_wayland_types_with_wtype_even_if_ydotool_exists(monkeypatch):
    box = Box(monkeypatch, {"wtype", "ydotool", "wl-copy"}, wayland=True)
    assert "wtype" in linux.can_paste()[1]
    assert linux.paste(TEXT) is True
    assert box.calls == [("wtype", ["-"], TEXT.encode())]


def test_wayland_ydotool_pastes_and_restores_the_clipboard(monkeypatch):
    box = Box(monkeypatch, {"ydotool", "wl-copy", "wl-paste"}, wayland=True)
    ok, why = linux.can_paste()
    assert ok and "ydotool" in why
    assert linux.paste(TEXT) is True
    assert box.tools_called() == ["wl-paste", "wl-copy", "ydotool", "wl-paste", "wl-copy"]
    assert box.calls[2][1] == ["key", "29:1", "47:1", "47:0", "29:0"]
    assert box.calls[1][2] == TEXT.encode() and box.clipboard == b"old"


def test_wayland_ydotool_does_not_clobber_a_newer_copy(monkeypatch):
    box = Box(monkeypatch, {"ydotool", "wl-copy", "wl-paste"}, wayland=True)

    def user_copies_meanwhile(argv, input=None, **kw):
        if argv[0] == "ydotool":
            box.clipboard = b"user copied this"
        return box.run(argv, input, **kw)

    monkeypatch.setattr(linux.subprocess, "run", user_copies_meanwhile)
    assert linux.paste(TEXT) is True
    assert box.clipboard == b"user copied this"


def test_gnome_without_virtual_keyboard_falls_through_to_ydotool(monkeypatch):
    box = Box(monkeypatch, {"wtype", "ydotool", "wl-copy"}, wayland=True, failing={"wtype"})
    assert linux.paste(TEXT) is True
    assert box.tools_called()[:2] == ["wtype", "wl-paste"] and "ydotool" in box.tools_called()


def test_ydotool_without_daemon_leaves_text_on_the_clipboard(monkeypatch):
    box = Box(monkeypatch, {"ydotool", "wl-copy"}, wayland=True, failing={"ydotool"})
    assert linux.paste(TEXT) is False
    assert box.clipboard == TEXT.encode()


def test_ydotool_alone_is_not_enough_to_type_unicode(monkeypatch):
    Box(monkeypatch, {"ydotool"}, wayland=True)
    assert linux.backends() == []


@pytest.mark.parametrize(
    ("wayland", "tools", "copier", "install"),
    [
        (True, {"wl-copy"}, "wl-copy", "wtype"),
        (False, {"xclip"}, "xclip", "xdotool"),
        (False, {"xsel"}, "xsel", "xdotool"),
    ],
)
def test_nothing_can_type_copies_to_the_clipboard(monkeypatch, wayland, tools, copier, install):
    box = Box(monkeypatch, tools, wayland=wayland)
    ok, why = linux.can_paste()
    assert not ok and install in why and "clipboard" in why
    assert linux.paste(TEXT) is False
    assert box.calls == [(copier, box.calls[0][1], TEXT.encode())]


def test_nothing_installed_does_not_crash(monkeypatch):
    box = Box(monkeypatch, set(), wayland=False)
    ok, why = linux.can_paste()
    assert not ok and "xdotool" in why and "no clipboard tool" in why
    assert linux.paste(TEXT) is False and box.calls == []


@pytest.mark.parametrize("error", [FileNotFoundError("gone"), subprocess.TimeoutExpired("xdotool", 5)])
def test_tool_that_cannot_run_counts_as_failure(monkeypatch, error):
    Box(monkeypatch, {"xdotool", "xclip"}, wayland=False)

    def boom(argv, **kw):
        raise error

    monkeypatch.setattr(linux.subprocess, "run", boom)
    assert linux.paste(TEXT) is False
