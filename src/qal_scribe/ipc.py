"""Local control socket so `qal-scribe toggle` can drive a running instance (workstream `platform`).

Contract: serve(on_toggle) starts a daemon thread; each `qal-scribe toggle [--translate]` calls on_toggle(mode)
with mode "dictate" | "translate". serve raises RuntimeError if another instance is already listening.
stop() closes the listener and removes the socket file (no-op when not serving). If on_toggle raises, the client
gets "error: <message>".
send_toggle(mode="dictate") -> process exit code (0 ok, 1 with a message on stderr).
Wire format: the client sends the mode and a newline; the server answers "ok\\n" or "error: ...\\n".
"""
import errno
import os
import socket
import stat
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from qal_scribe.events import Mode

TIMEOUT = 2.0
MODES = ("dictate", "translate")


def socket_path() -> Path:
    """$XDG_RUNTIME_DIR (private by spec) when set, else a per-user directory under the temp dir."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    folder = Path(runtime) if runtime else Path(tempfile.gettempdir()) / f"qal-scribe-{os.getuid()}"
    return folder / "qal-scribe.sock"


def _private_dir(folder: Path) -> None:
    """The socket's directory must be ours and closed to others, or another user could swap the socket."""
    folder.mkdir(mode=0o700, exist_ok=True)
    st = folder.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise RuntimeError(f"{folder} must be a directory owned by you with mode 0700; fix or remove it")


def _alive(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(TIMEOUT)
        try:
            probe.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            return False  # a leftover file from a process that died
    return True


def _bind(path: Path) -> socket.socket:
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            srv.bind(str(path))
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            if _alive(path):
                raise RuntimeError(f"qal-scribe is already running (control socket {path} is in use)") from None
            path.unlink(missing_ok=True)
            srv.bind(str(path))
        path.chmod(0o600)
        srv.listen()
    except BaseException:
        srv.close()
        raise
    return srv


def _handle(conn: socket.socket, on_toggle: Callable[[Mode], None]) -> None:
    conn.settimeout(TIMEOUT)
    try:
        mode = conn.recv(64).decode(errors="replace").strip()
        if not mode:  # liveness probe from a second instance
            return
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        on_toggle(mode)
        conn.sendall(b"ok\n")
    except Exception as e:  # noqa: BLE001 - one bad client or callback must not kill the listener
        with suppress(OSError):
            conn.sendall(f"error: {e}\n".encode())


def _accept_loop(srv: socket.socket, on_toggle: Callable[[Mode], None]) -> None:
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:  # listener closed
            return
        with conn:
            _handle(conn, on_toggle)


_servers: list[tuple[socket.socket, Path]] = []  # what serve() opened, for stop()


def serve(on_toggle: Callable[[Mode], None]) -> None:
    """Start a background listener; each `qal-scribe toggle [--translate]` calls on_toggle(mode)."""
    path = socket_path()
    _private_dir(path.parent)
    srv = _bind(path)
    _servers.append((srv, path))
    threading.Thread(target=_accept_loop, args=(srv, on_toggle), daemon=True, name="qal-scribe-ipc").start()


def stop() -> None:
    """Close the listener and remove the socket file, so the next start finds no stale socket."""
    while _servers:
        srv, path = _servers.pop()
        srv.close()
        path.unlink(missing_ok=True)


def send_toggle(mode: Mode = "dictate") -> int:
    path = socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(TIMEOUT)
            client.connect(str(path))
            client.sendall(f"{mode}\n".encode())
            reply = client.recv(256).decode(errors="replace").strip()
    except (FileNotFoundError, ConnectionRefusedError):
        print("qal-scribe is not running: start it first (run `qal-scribe` in a terminal), then toggle again",
              file=sys.stderr)
        return 1
    except OSError as e:
        print(f"could not reach qal-scribe at {path}: {e}", file=sys.stderr)
        return 1
    if reply != "ok":
        print(f"qal-scribe could not toggle: {reply or 'no answer'}", file=sys.stderr)
        return 1
    return 0
