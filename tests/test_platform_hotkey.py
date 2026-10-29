import sys
import types

import pytest

from qal_scribe import hotkey

real_input_monitoring_denied = hotkey.input_monitoring_denied


@pytest.fixture(autouse=True)
def no_real_permission_check(monkeypatch):
    monkeypatch.setattr(hotkey, "input_monitoring_denied", lambda: False)


def session(monkeypatch, **env):
    for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def fake_pynput(monkeypatch, start_error=None):
    class Listener:
        def __init__(self, on_press, on_release, daemon):
            self.on_press, self.on_release, self.started = on_press, on_release, False

        def start(self):
            if start_error:
                raise start_error
            self.started = True

        def stop(self):
            self.started = False

    keyboard = types.SimpleNamespace(Key=types.SimpleNamespace(alt_r="ALT_R", ctrl_r="CTRL_R"), Listener=Listener)
    monkeypatch.setitem(sys.modules, "pynput", types.SimpleNamespace(keyboard=keyboard))


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}, True),  # XWayland does not see global keys
        ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, True),  # sway from a console, XWayland running
        ({"XDG_SESSION_TYPE": "tty", "WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":0"}, True),
        ({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}, False),
        ({}, False),
    ],
)
def test_is_wayland(monkeypatch, env, expected):
    session(monkeypatch, **env)
    assert hotkey.is_wayland() is expected


def test_wayland_never_touches_pynput_and_explains(monkeypatch):
    session(monkeypatch, XDG_SESSION_TYPE="wayland")
    monkeypatch.setitem(sys.modules, "pynput", None)  # importing it would raise
    keys = hotkey.PushToTalk({"alt_r": "dictate"}, lambda m: None, lambda m: None)
    keys.start()
    keys.stop()
    assert "qal-scribe toggle" in keys.reason and "--translate" in keys.reason and not keys.listening


def test_no_x_display_sets_reason(monkeypatch):
    session(monkeypatch)
    monkeypatch.setitem(sys.modules, "pynput", None)
    keys = hotkey.PushToTalk({"alt_r": "dictate"}, lambda m: None, lambda m: None)
    keys.start()
    assert "unavailable" in keys.reason and "qal-scribe toggle" in keys.reason and not keys.listening


def test_listener_failing_to_start_sets_reason(monkeypatch):
    session(monkeypatch, DISPLAY=":0")
    fake_pynput(monkeypatch, start_error=RuntimeError("cannot connect to X server"))
    keys = hotkey.PushToTalk({"alt_r": "dictate"}, lambda m: None, lambda m: None)
    keys.start()
    assert "failed to start" in keys.reason and "cannot connect" in keys.reason and not keys.listening


def test_macos_input_monitoring_reason(monkeypatch):
    session(monkeypatch)
    fake_pynput(monkeypatch)
    monkeypatch.setattr(hotkey, "input_monitoring_denied", lambda: True)
    keys = hotkey.PushToTalk({"alt_r": "dictate"}, lambda m: None, lambda m: None)
    keys.start()
    assert "Input Monitoring" in keys.reason and keys._listener.started and keys.listening


def test_input_monitoring_check_uses_quartz(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Quartz", types.SimpleNamespace(CGPreflightListenEventAccess=lambda: False))
    assert real_input_monitoring_denied() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert real_input_monitoring_denied() is False  # Quartz is never imported off macOS


def test_working_setup_has_no_reason_and_one_key_at_a_time(monkeypatch):
    session(monkeypatch, DISPLAY=":0")
    fake_pynput(monkeypatch)
    events = []
    keys = hotkey.PushToTalk({"alt_r": "dictate", "ctrl_r": "translate"},
                             lambda m: events.append(("start", m)), lambda m: events.append(("stop", m)))
    keys.start()
    assert keys.reason is None and keys._listener.started
    press, release = keys._listener.on_press, keys._listener.on_release
    press("ALT_R")
    press("CTRL_R")  # ignored while Option is held
    release("CTRL_R")
    release("ALT_R")
    assert events == [("start", "dictate"), ("stop", "dictate")]
