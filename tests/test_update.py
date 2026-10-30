import io
import json
import sys
import urllib.request

import pytest

from qal_scribe import cli, doctor, update
from qal_scribe.config import Config


def pypi(monkeypatch, version="9.0.0"):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(json.dumps({"info": {"version": version}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_a_newer_release_is_reported_and_cached_for_a_day(monkeypatch):
    calls = pypi(monkeypatch)
    assert update.available() == "9.0.0"
    assert update.available() == "9.0.0" and len(calls) == 1  # second call reads the cache
    assert update.available(force=True) == "9.0.0" and len(calls) == 2


def test_same_older_or_prerelease_versions_are_not_updates(monkeypatch):
    monkeypatch.setattr(update, "__version__", "1.0.2")
    for version in ("1.0.2", "1.0.1", "1.1.0rc1"):
        pypi(monkeypatch, version)
        assert update.available(force=True) is None
    pypi(monkeypatch, "1.0.10")
    assert update.available(force=True) == "1.0.10"  # numeric, not string, comparison


def test_offline_or_a_broken_cache_is_silent():
    (update.cache_dir()).mkdir(parents=True)
    (update.cache_dir() / "update.json").write_text("not json")
    assert update.latest() is None and update.available() is None  # conftest's urlopen raises


def test_upgrade_command_keeps_a_cpu_torch_build(monkeypatch):
    monkeypatch.setattr(update.metadata, "version", lambda name: "2.15.0")
    assert update.upgrade_command() == ["uv", "tool", "install", "--python", "3.12", "qal-scribe@latest"]
    monkeypatch.setattr(update.metadata, "version", lambda name: "2.15.0+cpu")
    assert update.upgrade_command()[-3:] == ["qal-scribe@latest", "--torch-backend", "cpu"]


def test_update_command_runs_uv_for_a_uv_tool_install(monkeypatch, capsys):
    pypi(monkeypatch)
    ran = []
    monkeypatch.setattr(update, "is_uv_tool", lambda: True)
    monkeypatch.setattr("shutil.which", lambda name: "/bin/uv")
    monkeypatch.setattr("subprocess.call", lambda argv: ran.append(argv) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(["update"])
    assert e.value.code == 0 and ran == [update.upgrade_command()]
    assert "-> 9.0.0" in capsys.readouterr().out


def test_update_command_explains_other_installs_and_being_current(monkeypatch, capsys):
    pypi(monkeypatch)
    monkeypatch.setattr(update, "is_uv_tool", lambda: False)
    with pytest.raises(SystemExit) as e:
        cli.main(["update"])
    assert "not installed with `uv tool install`" in str(e.value.code)
    pypi(monkeypatch, update.__version__)
    cli.main(["update"])
    assert "is the newest version" in capsys.readouterr().out


def test_update_command_says_when_pypi_is_unreachable():
    with pytest.raises(SystemExit) as e:
        cli.main(["update"])
    assert "could not reach PyPI" in str(e.value.code)


def test_doctor_shows_an_available_update_unless_turned_off(monkeypatch):
    pypi(monkeypatch)
    check = doctor.check_updates(Config())
    assert check.status == "warn" and "9.0.0" in check.detail and check.fix == "run: qal-scribe update"
    assert doctor.check_updates(Config(check_updates=False)) is None


def test_is_uv_tool_reads_the_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert not update.is_uv_tool()
    (tmp_path / "uv-receipt.toml").write_text("")
    assert update.is_uv_tool()
