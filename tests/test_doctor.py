import re
import sys

import pytest

from qal_scribe import doctor, models
from qal_scribe.config import Config
from qal_scribe.doctor import Check
from qal_scribe.translate import TRANSLATION_MODELS

SPEC = models.MODELS["amh"]


def status(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_models_not_cached_warn_then_ok(hf_cache):
    cfg = Config()
    checks = doctor.check_models(cfg)
    assert [c.status for c in checks] == ["warn", "warn", "warn"]
    assert all("models download" in c.fix for c in checks)
    from qal_scribe.translate import TRANSLATION_MODELS

    nllb = TRANSLATION_MODELS["nllb-600m"]
    hf_cache(SPEC.repo, SPEC.revision, "model.safetensors")
    hf_cache(SPEC.repo, SPEC.revision, SPEC.lm_file)
    hf_cache(nllb.repo, nllb.revision, "pytorch_model.bin")
    assert [c.status for c in doctor.check_models(cfg)] == ["ok", "ok", "ok"]


def test_missing_lm_decoder_is_a_warning_with_the_fix(hf_cache, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    lm = next(c for c in doctor.check_models(Config()) if c.name == "Language model")
    assert lm.status == "warn" and "greedy" in lm.detail and "--python 3.12" in lm.fix


def test_models_cached_at_another_revision_do_not_count(hf_cache):
    hf_cache(SPEC.repo, "0" * 40, "model.safetensors")
    assert status(doctor.check_models(Config()), "Speech model") == "warn"


def test_disabled_features_are_not_nagged_about(hf_cache):
    checks = doctor.check_models(Config(lm=False, translate_key=""))
    assert status(checks, "Language model") == "ok" and status(checks, "Translation model") == "ok"


def test_translation_off_is_ok_not_a_warning(hf_cache):
    check = doctor.check_translation(Config(translation=False))
    assert check.status == "ok" and check.detail == "off (translation = false)"


def test_doctor_checks_the_chosen_translation_model(hf_cache):
    madlad = TRANSLATION_MODELS["madlad-3b"]
    cfg = Config(translation_model="madlad-3b")
    check = doctor.check_translation(cfg)
    assert check.status == "warn" and madlad.repo in check.detail and "11.8 GB" in check.detail
    hf_cache(madlad.repo, madlad.revision, "model.safetensors")
    assert doctor.check_translation(cfg).status == "ok"


def test_unknown_translation_model_is_a_config_problem():
    assert "nllb-600m, nllb-1.3b, madlad-3b" in doctor.config_problem(Config(translation_model="gpt"))


def test_disk_need_grows_with_the_translation_model(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 10_000_000_000})())
    assert doctor.check_disk(Config()).status == "ok"
    assert doctor.check_disk(Config(translation_model="madlad-3b")).status == "warn"
    assert doctor.check_disk(Config(translation_model="madlad-3b", translation=False)).status == "ok"


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


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_config_problem_restore_delay_must_be_seconds(delay):
    assert "0 or more" in doctor.config_problem(Config(restore_delay=delay))


def test_config_problem_empty_dictate_key():
    assert "dictate_key is empty" in doctor.config_problem(Config(dictate_key=""))


@pytest.mark.parametrize("device", ["gpu", "cuda" if sys.platform == "darwin" else "mps", "cuda:7"])
def test_config_problem_device_that_is_not_here(device):
    if device.startswith("cuda") and doctor.available_devices().count("cuda"):
        pytest.skip("this machine has CUDA")
    assert f"device {device!r} is not available here" in doctor.config_problem(Config(device=device))
    assert doctor.config_problem(Config(device="cpu")) is None


def test_config_problem_unreadable_words_file(tmp_path):
    words = tmp_path / "words.txt"
    words.write_text("ኤፒአይ => API\n", encoding="utf-16")
    assert "not UTF-8" in doctor.config_problem(Config(words_file=words))
    assert str(tmp_path) in doctor.config_problem(Config(words_file=tmp_path))


def sounddevice_or_skip():
    try:
        import sounddevice
    except Exception as e:  # noqa: BLE001 - import raises PortAudioError where there is no audio server (CI)
        pytest.skip(f"sounddevice unavailable: {e}")
    return sounddevice


def test_microphone_without_audio_server(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise RuntimeError("PulseAudio_Initialize: Can't connect to server")  # stands in for PortAudioError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    mic = doctor.check_microphone()
    assert mic.status == "fail" and "audio system unavailable" in mic.detail


def test_microphone(monkeypatch):
    sd = sounddevice_or_skip()
    monkeypatch.setattr(sd, "query_devices", lambda device=None, kind=None: {"name": "Fake mic"})
    assert doctor.check_microphone().detail == "Fake mic"

    def none(device=None, kind=None):
        raise sd.PortAudioError("Error querying device -1")

    monkeypatch.setattr(sd, "query_devices", none)
    mic = doctor.check_microphone()
    assert mic.status == "fail" and "microphone" in mic.fix


def test_configured_microphone_is_resolved_like_the_recorder_does(monkeypatch):
    sd = sounddevice_or_skip()
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


def test_paste_failure_on_unsupported_platforms_shows_its_reason_as_the_fix(monkeypatch):
    from qal_scribe import paste

    monkeypatch.setattr(sys, "platform", "win32")
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
    assert check.status == "fail" and check.url == doctor.PANES["Input Monitoring"]
    assert f'open "{check.url}"' in check.fix and "Privacy_ListenEvent" in check.url


@pytest.mark.parametrize(("platform", "expected"), [("darwin", "ok"), ("linux", "warn"), ("win32", "fail")])
def test_system(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    assert doctor.check_system().status == expected


def test_device_names_what_dictation_will_use():
    check = doctor.check_device(Config(device="cpu"))
    assert check.status == "ok" and check.detail.startswith("cpu (torch ")


@pytest.mark.parametrize(("status", "expected"), [(0, "ok"), (1, "fail"), (2, "fail"), (3, "ok")])
def test_microphone_permission(monkeypatch, status, expected):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(doctor, "microphone_permission", lambda: status)
    check = doctor.check_microphone_permission()
    assert check.status == expected
    assert ("Privacy_Microphone" in check.fix and check.url == doctor.PANES["Microphone"]) == (expected == "fail")


def test_microphone_permission_is_macos_only(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert doctor.check_microphone_permission() is None


def test_disk_warns_when_low(monkeypatch, hf_cache):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 1_000_000_000})())
    assert doctor.check_disk().status == "warn"


def run_with(monkeypatch, tmp_path, **checks):
    ok = Check("x", "ok", "fine")
    defaults = {"check_system": ok, "check_device": ok, "check_microphone": ok, "check_microphone_permission": None,
                "check_paste": ok, "check_input_monitoring": None, "check_disk": ok}
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


def linux(monkeypatch, os_id="ubuntu", like="debian", wayland=False):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(doctor.platform, "freedesktop_os_release", lambda: {"ID": os_id, "ID_LIKE": like})
    monkeypatch.setattr(doctor, "is_wayland", lambda: wayland)


@pytest.mark.parametrize(("os_id", "like", "expected"), [
    ("ubuntu", "debian", "sudo apt install libportaudio2 xclip"),
    ("pop", "ubuntu debian", "sudo apt install libportaudio2 xclip"),
    ("fedora", "", "sudo dnf install portaudio xclip"),
    ("nobara", "fedora", "sudo dnf install portaudio xclip"),
    ("endeavouros", "arch", "sudo pacman -S portaudio xclip"),
    ("nixos", "", "install: portaudio, xclip"),
])
def test_linux_install_picks_the_package_manager_and_names(monkeypatch, os_id, like, expected):
    linux(monkeypatch, os_id, like)
    assert doctor.linux_install("portaudio", "xclip") == expected


def test_linux_install_without_os_release(monkeypatch):
    monkeypatch.setattr(doctor.platform, "freedesktop_os_release", lambda: (_ for _ in ()).throw(OSError()))
    assert doctor.linux_install("build") == "install: a C compiler and the Python headers"


@pytest.mark.parametrize(("term", "name"), [("iTerm.app", "iTerm2"), ("Apple_Terminal", "Terminal"),
                                            ("something", "your terminal app"), (None, "your terminal app")])
def test_terminal_name(monkeypatch, term, name):
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    if term:
        monkeypatch.setenv("TERM_PROGRAM", term)
    assert doctor.terminal_name() == name


def test_macos_permission_fixes_open_the_pane_and_name_the_terminal(monkeypatch):
    from qal_scribe import paste

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.setattr(paste, "can_paste", lambda: (False, "x"))
    check = doctor.check_paste()
    assert check.name == "Accessibility" and check.detail == "iTerm2 cannot paste yet" and check.url
    assert check.fix == f'open "{doctor.PANES["Accessibility"]}", turn on iTerm2, then quit and reopen iTerm2'
    assert doctor.PANES["Accessibility"].endswith("Privacy_Accessibility")


@pytest.mark.parametrize(("wayland", "expected"), [
    (True, ["sudo apt install wtype", "sudo apt install ydotool wl-clipboard", "ydotoold"]),
    (False, ["sudo apt install xdotool xclip"]),
])
def test_linux_text_insertion_fix_is_a_command(monkeypatch, wayland, expected):
    from qal_scribe import paste

    linux(monkeypatch, wayland=wayland)
    monkeypatch.setattr(paste, "can_paste", lambda: (False, "cannot type text"))
    check = doctor.check_paste()
    assert check.name == "Text insertion" and all(part in check.fix for part in expected)


def test_linux_hotkeys_row(monkeypatch):
    linux(monkeypatch, wayland=True)
    check = doctor.check_hotkeys()
    assert check.status == "warn" and "qal-scribe toggle --translate" in check.fix
    linux(monkeypatch)
    monkeypatch.setitem(sys.modules, "pynput", None)  # import fails like a missing X display
    monkeypatch.delitem(sys.modules, "pynput.keyboard", raising=False)
    check = doctor.check_hotkeys()
    assert check.status == "fail" and "X11 or Wayland" in check.fix
    monkeypatch.setattr(sys, "platform", "darwin")
    assert doctor.check_hotkeys() is None


def missing_sounddevice(monkeypatch, error):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_portaudio_fix_depends_on_the_platform(monkeypatch):
    missing_sounddevice(monkeypatch, OSError("PortAudio library not found"))
    linux(monkeypatch, "fedora", "")
    assert doctor.check_microphone().fix == "sudo dnf install portaudio"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "uv tool install --python 3.12 --reinstall qal-scribe" == doctor.check_microphone().fix


def test_audio_server_fix_on_linux(monkeypatch):
    missing_sounddevice(monkeypatch, RuntimeError("Can't connect to server"))
    linux(monkeypatch)
    assert doctor.check_microphone().fix == "systemctl --user start pipewire pipewire-pulse"


def test_config_and_model_fixes_are_commands(tmp_path, hf_cache):
    bad = tmp_path / "config.toml"
    bad.write_text('language = "xx"\n')
    assert str(bad) in doctor.check_config(bad)[0].fix
    assert all("qal-scribe setup" in c.fix for c in doctor.check_models(Config()))
