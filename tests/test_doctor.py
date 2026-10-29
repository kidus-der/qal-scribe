import re
import sys

import pytest

from qal_scribe import doctor, models
from qal_scribe.config import Config
from qal_scribe.doctor import Check

SPEC = models.MODELS["amh"]


def status(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_models_not_cached_warn_then_ok(hf_cache):
    cfg = Config()
    checks = doctor.check_models(cfg)
    assert [c.status for c in checks] == ["warn", "warn", "warn"]
    assert all("models download" in c.fix for c in checks)
    from qal_scribe.translate import Translator

    hf_cache(SPEC.repo, SPEC.revision, "model.safetensors")
    hf_cache(SPEC.repo, SPEC.revision, SPEC.lm_file)
    hf_cache(Translator.MODEL, Translator.REVISION, "pytorch_model.bin")
    assert [c.status for c in doctor.check_models(cfg)] == ["ok", "ok", "ok"]


def test_models_cached_at_another_revision_do_not_count(hf_cache):
    hf_cache(SPEC.repo, "0" * 40, "model.safetensors")
    assert status(doctor.check_models(Config()), "Speech model") == "warn"


def test_disabled_features_are_not_nagged_about(hf_cache):
    checks = doctor.check_models(Config(lm=False, translate_key=""))
    assert status(checks, "Language model") == "ok" and status(checks, "Translation model") == "ok"


def test_check_config(tmp_path):
    good, cfg = doctor.check_config(tmp_path / "missing.toml")
    assert good.status == "ok" and cfg == Config()
    bad = tmp_path / "config.toml"
    bad.write_text("nonsense = 1\n")
    check, cfg = doctor.check_config(bad)
    assert check.status == "fail" and "nonsense" in check.detail and cfg == Config()
    bad.write_text('language = "xx"\n')
    check, cfg = doctor.check_config(bad)
    assert "no model for language" in check.detail and cfg == Config()  # the other checks must not see "xx"
    bad.write_text('restore_delay = "0.5"\n')
    check, cfg = doctor.check_config(bad)
    assert check.status == "fail" and "restore_delay must be a number" in check.detail and cfg == Config()


def test_config_problem_same_key_twice():
    assert "both" in doctor.config_problem(Config(dictate_key="alt_r", translate_key="alt_r"))
    assert doctor.config_problem(Config(translate_key="")) is None


def test_config_problem_negative_restore_delay():
    assert "negative" in doctor.config_problem(Config(restore_delay=-1))


def test_microphone(monkeypatch):
    sd = pytest.importorskip("sounddevice")
    monkeypatch.setattr(sd, "query_devices", lambda device=None, kind=None: {"name": "Fake mic"})
    assert doctor.check_microphone().detail == "Fake mic"

    def none(device=None, kind=None):
        raise sd.PortAudioError("Error querying device -1")

    monkeypatch.setattr(sd, "query_devices", none)
    mic = doctor.check_microphone()
    assert mic.status == "fail" and "microphone" in mic.fix


def test_configured_microphone_is_resolved_like_the_recorder_does(monkeypatch):
    sd = pytest.importorskip("sounddevice")
    devices = [{"name": "MacBook Pro Microphone", "max_input_channels": 1},
               {"name": "MacBook Pro Speakers", "max_input_channels": 0},
               {"name": "USB Mic", "max_input_channels": 2}]

    def query(device=None, kind=None):
        if device is None and kind is None:
            return devices
        if isinstance(device, int):
            return devices[device]
        for d in devices:
            if device in d["name"] and d["max_input_channels"]:
                return d
        raise ValueError(f"No input device matching {device!r}")

    monkeypatch.setattr(sd, "query_devices", query)
    assert doctor.check_microphone("USB").detail == "USB Mic"
    assert doctor.check_microphone("2").detail == "USB Mic"  # a number in the config is a device index
    mic = doctor.check_microphone("Yeti")
    assert mic.status == "fail" and "Yeti" in mic.detail
    assert "MacBook Pro Microphone, USB Mic" in mic.fix and "Speakers" not in mic.fix


def test_paste_failure_shows_its_reason_as_the_fix(monkeypatch):
    from qal_scribe import paste

    monkeypatch.setattr(paste, "can_paste", lambda: (False, "grant the permission"))
    check = doctor.check_paste()
    assert (check.status, check.fix) == ("fail", "grant the permission")
    monkeypatch.setattr(paste, "can_paste", lambda: (True, "fine"))
    assert doctor.check_paste().status == "ok"


def test_input_monitoring_is_macos_only(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert doctor.check_input_monitoring() is None


def test_input_monitoring_denied(monkeypatch):
    Quartz = pytest.importorskip("Quartz")
    monkeypatch.setattr(Quartz, "CGPreflightListenEventAccess", lambda: False)
    check = doctor.check_input_monitoring()
    assert check.status == "fail" and "Input Monitoring" in check.fix


@pytest.mark.parametrize(("platform", "expected"), [("darwin", "ok"), ("linux", "warn"), ("win32", "fail")])
def test_system(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    assert doctor.check_system().status == expected


def test_device_that_does_not_exist_fails():
    assert doctor.check_device(Config(device="cuda:7" if sys.platform == "darwin" else "mps")).status == "fail"
    assert doctor.check_device(Config(device="cpu")).status == "ok"


def test_disk_warns_when_low(monkeypatch, hf_cache):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 1_000_000_000})())
    assert doctor.check_disk().status == "warn"


def run_with(monkeypatch, tmp_path, **checks):
    ok = Check("x", "ok", "fine")
    defaults = {"check_system": ok, "check_device": ok, "check_microphone": ok, "check_paste": ok,
                "check_input_monitoring": None, "check_disk": ok}
    for name, result in {**defaults, **checks}.items():
        monkeypatch.setattr(doctor, name, lambda *_, result=result: result)
    return doctor.run(tmp_path / "config.toml")


def test_exit_code_is_1_only_when_dictation_is_blocked(monkeypatch, tmp_path, hf_cache, capsys):
    assert run_with(monkeypatch, tmp_path) == 0  # models missing is only a warning
    assert "Ready to dictate" in capsys.readouterr().out
    assert run_with(monkeypatch, tmp_path, check_microphone=Check("Microphone", "fail", "none", "plug one in")) == 1
    out = capsys.readouterr().out
    assert "Dictation is blocked" in out and "plug one in" in out


@pytest.mark.parametrize("text", ['language = "xx"', 'language = "am"', 'restore_delay = "fast"', "words_file = 5"])
def test_bad_config_renders_the_table_instead_of_crashing(monkeypatch, tmp_path, hf_cache, capsys, text):
    (tmp_path / "config.toml").write_text(text + "\n")
    assert run_with(monkeypatch, tmp_path) == 1
    out = capsys.readouterr().out
    assert re.search(r"Config\s+│\s+FAIL", out) and "Dictation is blocked" in out
