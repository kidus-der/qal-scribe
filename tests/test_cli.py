import re
import sys
from dataclasses import fields
from importlib.resources import files

import pytest

from qal_scribe import cli, config, models


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never read or write the real ~/.config/qal-scribe."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def exit_message(argv) -> str:
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    return str(e.value.code)


def test_models_lists_amharic(capsys):
    cli.main(["models"])
    assert "snapwre/hohe-asr-amharic" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "qal-scribe" in capsys.readouterr().out


def test_plain_format_ready():
    from qal_scribe import events as ev
    from qal_scribe.plain import format_event

    line = format_event(ev.Ready("mps", "m", "amh", True, {"Right Option": "dictate"}))
    assert "hold Right Option to dictate" in line and "LM on" in line


def test_about_prints_the_packaged_notice(capsys):
    cli.main(["about"])
    out = capsys.readouterr().out
    assert cli.__version__ in out and "CC-BY-NC-4.0" in out


def test_config_prints_path_and_init_refuses_to_overwrite(tmp_path, capsys):
    path = tmp_path / "sub" / "config.toml"
    cli.main(["--config", str(path), "config"])
    assert capsys.readouterr().out.strip() == str(path)
    cli.main(["--config", str(path), "config", "--init"])
    original = path.read_text(encoding="utf-8")
    assert original.startswith("# qal-scribe settings")
    path.write_text("lm = false\n")
    assert "already exists" in exit_message(["--config", str(path), "config", "--init"])
    assert path.read_text() == "lm = false\n"


def test_default_config_template_lists_every_setting():
    """When you add a field to Config, add it to default_config.toml too."""
    template = files("qal_scribe").joinpath("default_config.toml").read_text(encoding="utf-8")
    keys = re.findall(r"^# ([a-z_]+) = ", template, re.MULTILINE)
    assert sorted(keys) == sorted(f.name for f in fields(config.Config))


def test_uncommented_template_is_a_valid_config(tmp_path):
    template = files("qal_scribe").joinpath("default_config.toml").read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(re.sub(r"^# (?=[a-z_]+ = )", "", template, flags=re.MULTILINE), encoding="utf-8")
    assert config.load(path).language == "amh"


def test_bad_config_is_one_friendly_line(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('langauge = "amh"\n')
    message = exit_message(["--config", str(path), "--plain"])
    assert "bad config" in message and "langauge" in message and "Traceback" not in message
    path.write_text("lm = \n")
    assert "bad config" in exit_message(["--config", str(path), "--plain"])


def test_explicit_config_that_does_not_exist_is_an_error(tmp_path):
    assert "no config file" in exit_message(["--config", str(tmp_path / "typo.toml"), "models", "download"])


def test_unsupported_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert "not supported" in exit_message(["--plain"])


def test_unknown_hotkey_name_is_friendly(tmp_path):
    pytest.importorskip("pynput.keyboard", exc_type=ImportError)  # needs a display on Linux
    path = tmp_path / "config.toml"
    path.write_text('dictate_key = "option_r"\n')
    assert "option_r" in exit_message(["--config", str(path), "--plain"])


def test_models_download_fetches_speech_model(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: calls.append((spec.language, with_lm)) or tmp_path)
    cli.main(["models", "download"])
    assert calls == [("amh", True)] and str(tmp_path) in capsys.readouterr().out


def test_models_download_translation(monkeypatch, tmp_path, capsys):
    import huggingface_hub

    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: f"fetched {repo} {kw['revision'][:7]}")
    cli.main(["models", "download", "--translation"])
    assert "fetched facebook/nllb-200-distilled-600M" in capsys.readouterr().out


def test_models_download_unknown_language():
    assert "no model for language 'xx'" in exit_message(["models", "download", "xx"])


def test_download_failure_is_friendly(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(spec, with_lm=False):
        raise LocalEntryNotFoundError("cannot reach the Hub")

    monkeypatch.setattr(models, "download", offline)
    message = exit_message(["models", "download"])
    assert "cannot reach the Hub" in message and "qal-scribe models download" in message
