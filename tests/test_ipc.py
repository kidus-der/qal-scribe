import os
import queue
import shutil
import socket
import tempfile

import pytest

from qal_scribe import cli, ipc


@pytest.fixture
def runtime(monkeypatch):
    # Short path: AF_UNIX paths are limited to ~104 bytes and pytest's tmp_path can exceed that on macOS.
    folder = tempfile.mkdtemp(prefix="qs")
    monkeypatch.setenv("XDG_RUNTIME_DIR", folder)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def test_round_trip_passes_the_mode(runtime):
    got = queue.Queue()
    ipc.serve(got.put)
    assert ipc.send_toggle("translate") == 0
    assert ipc.send_toggle() == 0
    assert [got.get(timeout=2), got.get(timeout=2)] == ["translate", "dictate"]


def test_socket_is_private(runtime):
    ipc.serve(lambda mode: None)
    assert (os.stat(ipc.socket_path()).st_mode & 0o777) == 0o600


def test_no_instance_is_a_helpful_error(runtime, capsys):
    assert ipc.send_toggle() == 1
    assert "not running" in capsys.readouterr().err


def test_stale_socket_is_cleaned_up(runtime):
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(str(ipc.socket_path()))
    stale.close()  # file stays behind, nobody listens
    assert ipc.send_toggle() == 1
    got = queue.Queue()
    ipc.serve(got.put)
    assert ipc.send_toggle() == 0 and got.get(timeout=2) == "dictate"


def test_second_instance_is_refused(runtime):
    ipc.serve(lambda mode: None)
    with pytest.raises(RuntimeError, match="already running"):
        ipc.serve(lambda mode: None)
    assert ipc.send_toggle() == 0  # the first instance is unharmed by the probe


def test_failing_callback_is_reported_and_listener_survives(runtime, capsys):
    calls = []

    def on_toggle(mode):
        calls.append(mode)
        if len(calls) == 1:
            raise RuntimeError("mic busy")

    ipc.serve(on_toggle)
    assert ipc.send_toggle() == 1
    assert "mic busy" in capsys.readouterr().err
    assert ipc.send_toggle() == 0


def test_unknown_mode_is_rejected(runtime):
    ipc.serve(lambda mode: pytest.fail("must not be called"))
    with socket.socket(socket.AF_UNIX) as c:
        c.connect(str(ipc.socket_path()))
        c.sendall(b"rm -rf\n")
        assert c.recv(100).startswith(b"error: unknown mode")


def test_fallback_dir_is_per_user_and_private(monkeypatch):
    base = tempfile.mkdtemp(prefix="qs")
    try:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setattr(tempfile, "gettempdir", lambda: base)
        got = queue.Queue()
        ipc.serve(got.put)
        folder = ipc.socket_path().parent
        assert folder.name == f"qal-scribe-{os.getuid()}" and (folder.stat().st_mode & 0o777) == 0o700
        assert ipc.send_toggle() == 0 and got.get(timeout=2) == "dictate"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_open_directory_is_refused(runtime):
    os.chmod(runtime, 0o755)
    with pytest.raises(RuntimeError, match="0700"):
        ipc.serve(lambda mode: None)


@pytest.mark.parametrize(("argv", "mode"), [(["toggle"], "dictate"), (["toggle", "--translate"], "translate")])
def test_cli_toggle_passes_mode(monkeypatch, argv, mode):
    seen = []
    monkeypatch.setattr(ipc, "send_toggle", lambda m: seen.append(m) or 0)
    with pytest.raises(SystemExit) as exit_:
        cli.main(argv)
    assert exit_.value.code == 0 and seen == [mode]
