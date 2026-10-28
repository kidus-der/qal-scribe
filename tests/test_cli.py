import pytest

from qal_scribe import cli


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
