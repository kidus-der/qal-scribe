"""Whether PyPI has a newer qal-scribe, and the command that installs it.

Contract: available(force=False) -> "X.Y.Z" when PyPI has a newer release than the running one, else None. It asks
PyPI at most once a day (the answer is cached in cache_dir()/update.json; force skips the cache) and never raises:
offline or any error means None. upgrade_command() -> the argv that upgrades this install. is_uv_tool() -> whether
this copy was installed with `uv tool install`.
"""
import json
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path

from qal_scribe import __version__
from qal_scribe.models import cache_dir

URL = "https://pypi.org/pypi/qal-scribe/json"
DAY = 24 * 3600


def parse(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:  # pre-releases and dev builds ("1.1.0rc1", "1.0.2.dev3") never prompt
        return None


def latest(force: bool = False) -> str | None:
    path = cache_dir() / "update.json"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if not force and time.time() - cached["checked"] < DAY:
            return cached["latest"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        with urllib.request.urlopen(URL, timeout=3) as response:
            version = json.load(response)["info"]["version"]
    except Exception:  # noqa: BLE001 - offline, DNS, TLS, a changed PyPI reply: no update notice, never a crash
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked": time.time(), "latest": version}), encoding="utf-8")
    except OSError:
        pass
    return version


def available(force: bool = False) -> str | None:
    new = latest(force)
    mine, theirs = parse(__version__), parse(new) if new else None
    return new if mine and theirs and theirs > mine else None


def is_uv_tool() -> bool:
    return (Path(sys.prefix) / "uv-receipt.toml").exists()  # uv writes this into every tool environment


def upgrade_command() -> list[str]:
    try:
        cpu_torch = "+cpu" in metadata.version("torch")
    except metadata.PackageNotFoundError:
        cpu_torch = False
    # @latest, not `uv tool upgrade`: that does nothing for an install pinned to a version (qal-scribe==1.0.1)
    command = ["uv", "tool", "install", "--python", "3.12", "qal-scribe@latest"]
    if cpu_torch:  # uv does not remember --torch-backend, and an upgrade could pull the multi-GB CUDA build
        command += ["--torch-backend", "cpu"]
    return command
