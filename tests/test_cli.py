import re
import sys
from dataclasses import fields
from importlib.resources import files

import pytest

from qal_scribe import cli, config, models


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, hf_cache):
    """Never touch the real ~/.config/qal-scribe or Hugging Face cache, and never start a UI: a guard that
    regresses must fail fast, not load weights, open the microphone and block forever."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def no_ui(cfg, *args, **kwargs):
        raise AssertionError("UI started")

    monkeypatch.setattr("qal_scribe.plain.run", no_ui)
    monkeypatch.setattr("qal_scribe.app.run", no_ui)


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


def test_plain_ready_without_hotkeys_mentions_translate_only_when_it_is_on():
    from qal_scribe import events as ev
    from qal_scribe.plain import format_event

    ready = ev.Ready("mps", "m", "amh", True, {})
    assert "toggle --translate" in format_event(ready) and "toggle --translate" not in format_event(ready, False)


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
    path.write_text('restore_delay = "0.5"\n')
    assert "restore_delay must be a number" in exit_message(["--config", str(path)])


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
    path = tmp_path / "config.toml"
    path.write_text('translation_model = "nllb-1.3b"\n')
    cli.main(["--config", str(path), "models", "download", "--translation"])
    assert "fetched facebook/nllb-200-distilled-1.3B" in capsys.readouterr().out


def test_models_download_unknown_language():
    assert "no model for language 'xx'" in exit_message(["models", "download", "xx"])


def test_download_failure_is_friendly(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(spec, with_lm=False):
        raise LocalEntryNotFoundError("cannot reach the Hub")

    monkeypatch.setattr(models, "download", offline)
    message = exit_message(["models", "download"])
    assert "cannot reach the Hub" in message and "qal-scribe models download" in message


def test_first_run_download_failure_is_friendly_before_the_ui_starts(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(spec, with_lm=False):
        raise LocalEntryNotFoundError("cannot reach the Hub")

    monkeypatch.setattr(models, "download", offline)
    message = exit_message(["--plain"])  # the TUI downloads on its setup screen instead
    assert "cannot reach the Hub" in message and "qal-scribe models download" in message


def test_the_tui_does_not_download_before_it_opens(monkeypatch):
    seen = []
    monkeypatch.setattr(models, "download", lambda *a, **k: seen.append("speech"))
    monkeypatch.setattr("qal_scribe.translate.download", lambda *a, **k: seen.append("translation"))
    monkeypatch.setattr("qal_scribe.app.run", lambda cfg, path=None, setup=None: seen.append((path.name, setup)))
    cli.main([])
    cli.main(["setup"])
    assert seen == [("config.toml", None), ("config.toml", True)]


def test_setup_subcommand_parses():
    assert cli.build_parser().parse_args(["setup"]).command == "setup"


def test_models_are_fetched_before_the_plain_output_only_when_missing(monkeypatch, hf_cache):
    from qal_scribe.translate import TRANSLATION_MODELS

    spec = models.MODELS["amh"]
    nllb = TRANSLATION_MODELS["nllb-600m"]
    downloads, started = [], []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: downloads.append(with_lm))
    monkeypatch.setattr("qal_scribe.translate.download", lambda model: downloads.append(model))
    monkeypatch.setattr("qal_scribe.plain.run", started.append)
    cli.main(["--plain"])
    assert downloads == [True, "nllb-600m"] and len(started) == 1  # nothing cached
    hf_cache(spec.repo, spec.revision, "model.safetensors")
    hf_cache(nllb.repo, nllb.revision, "pytorch_model.bin")
    cli.main(["--plain"])
    assert downloads == [True, "nllb-600m", True]  # the language model is still missing
    hf_cache(spec.repo, spec.revision, spec.lm_file)
    cli.main(["--plain"])
    assert downloads == [True, "nllb-600m", True] and len(started) == 3


@pytest.mark.parametrize("config, flags", [('translate_key = ""\n', []), ("translation = false\n", []),
                                           ("", ["--no-translate"])])
def test_no_translation_means_no_translation_download(monkeypatch, tmp_path, config, flags):
    downloads, started = [], []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: None)
    monkeypatch.setattr("qal_scribe.translate.download", lambda model: downloads.append(model))
    monkeypatch.setattr("qal_scribe.plain.run", started.append)
    path = tmp_path / "config.toml"
    path.write_text(config)
    cli.main(["--plain", *flags, "--config", str(path)])
    assert downloads == [] and started[0].translation == (not flags and "false" not in config)


def test_the_chosen_translation_model_is_the_one_fetched(monkeypatch, tmp_path):
    downloads = []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: None)
    monkeypatch.setattr("qal_scribe.translate.download", lambda model: downloads.append(model))
    monkeypatch.setattr("qal_scribe.plain.run", lambda cfg: None)
    path = tmp_path / "config.toml"
    path.write_text('translation_model = "madlad-3b"\n')
    cli.main(["--plain", "--config", str(path)])
    assert downloads == ["madlad-3b"]


def test_a_failed_translation_download_still_starts_dictation(monkeypatch, capsys):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(model):
        raise LocalEntryNotFoundError("cannot reach the Hub")

    started = []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: None)
    monkeypatch.setattr("qal_scribe.translate.download", offline)
    monkeypatch.setattr("qal_scribe.plain.run", started.append)
    cli.main(["--plain"])
    err = capsys.readouterr().err
    assert len(started) == 1 and "cannot reach the Hub" in err and "models download --translation" in err


def test_downloads_let_ctrl_c_through_and_restore_the_handler(monkeypatch):
    import signal

    seen = []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: seen.append(signal.getsignal(signal.SIGINT)))
    before = signal.getsignal(signal.SIGINT)
    cli.main(["models", "download"])
    assert seen == [signal.SIG_DFL] and signal.getsignal(signal.SIGINT) is before


def test_bad_device_or_words_file_stops_before_the_model_loads(tmp_path, monkeypatch):
    def no_download(spec, with_lm=False):
        raise AssertionError("downloaded")

    monkeypatch.setattr(models, "download", no_download)
    path = tmp_path / "config.toml"
    path.write_text('device = "gpu"\n')
    assert "device 'gpu' is not available here" in exit_message(["--config", str(path), "--plain"])
    words = tmp_path / "words.txt"
    words.write_text("ኤፒአይ => API\n", encoding="utf-16")
    path.write_text(f'words_file = "{words}"\n')
    for argv in (["--plain"], ["transcribe", write_wav(tmp_path / "a.wav", 16000)]):
        message = exit_message(["--config", str(path), *argv])
        assert "bad config" in message and "words.txt: not UTF-8" in message and "Traceback" not in message


def test_plain_format_says_when_recording_starts_and_stops():
    from qal_scribe import events as ev
    from qal_scribe.plain import format_event

    assert format_event(ev.RecordingStarted("dictate")) == "● recording (dictate)"
    assert format_event(ev.RecordingStopped("translate", 2.04)) == "… 2.0s recorded, translating"
    assert format_event(ev.Partial("draft")) is None  # too chatty for a log
    assert format_event(ev.Transcribed("dictate", "", "", 0.4, 0.05, False)).endswith("0.05s] (nothing recognised)")
    assert format_event(ev.Transcribed("dictate", "ሰላም", "", 1, 0.1, False)).endswith("] ሰላም (not pasted)")


def test_help_says_how_to_start_and_subcommands_describe_themselves(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "Run with no command to start dictating" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["doctor", "--help"])
    assert "check permissions" in capsys.readouterr().out


def test_models_lists_aligned_columns_and_the_translation_model(capsys):
    cli.main(["models"])
    lines = capsys.readouterr().out.splitlines()
    table = lines[: len(models.MODELS) + 2 + 2]
    assert "facebook/nllb-200-distilled-600M" in table[-3] and "not downloaded" in table[-3]
    assert "google/madlad400-3b-mt" in table[-1] and "Apache-2.0" in table[-1]
    assert len({re.search(r"CC-BY|Apache", line).start() for line in table[1:]}) == 1  # one licence column
    assert lines[-1].startswith("download one with: qal-scribe models download LANG")


def test_doctor_with_a_missing_explicit_config_fails_like_the_other_commands(tmp_path):
    assert "no config file" in exit_message(["--config", str(tmp_path / "typo.toml"), "doctor"])


def test_download_failure_drops_the_hubs_got_prefix(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(spec, with_lm=False):
        raise LocalEntryNotFoundError("Got: ConnectError: [Errno 61] Connection refused\nAn error happened ...")

    monkeypatch.setattr(models, "download", offline)
    assert "\n  ConnectError: [Errno 61] Connection refused\n" in exit_message(["models", "download"])


# --- transcribe -----------------------------------------------------------------------------------------------


@pytest.fixture
def fake_engine(monkeypatch, hf_cache):
    """transcribe with a stand-in Engine that records the audio it was given; the models count as downloaded."""
    import numpy as np

    from qal_scribe import engine

    spec = models.MODELS["amh"]
    for f in ("model.safetensors", spec.lm_file):
        hf_cache(spec.repo, spec.revision, f)
    heard = []

    class Engine:
        device = "cpu"

        def __init__(self, spec, device=None, use_lm=True):
            pass

        def load(self, on_progress=None):
            pass

        def transcribe(self, audio, sentences=False):
            heard.append(np.asarray(audio))
            return engine.Result("ሞዴል ይሰራል", 0.1, ("ሞዴል", "ይሰራል") if sentences else ())

    monkeypatch.setattr(engine, "Engine", Engine)
    monkeypatch.setattr(cli, "quiet_progress_bars", lambda: None)  # process-wide; keep it out of other tests
    return heard


def write_wav(path, rate, channels=1, seconds=1.0):
    import numpy as np
    import soundfile as sf

    t = np.arange(int(rate * seconds)) / rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, np.stack([tone] * channels, axis=1) if channels > 1 else tone, rate)
    return str(path)


def test_transcribe_applies_the_words_file(tmp_path, capsys, fake_engine):
    (tmp_path / "xdg" / "qal-scribe").mkdir(parents=True)
    (tmp_path / "xdg" / "qal-scribe" / "words.txt").write_text("ሞዴል => MODEL\n", encoding="utf-8")
    cli.main(["transcribe", write_wav(tmp_path / "a.wav", 16000)])
    assert capsys.readouterr().out == "MODEL ይሰራል\n"


def test_transcribe_takes_stereo_and_other_sample_rates(tmp_path, capsys, fake_engine):
    import numpy as np

    cli.main(["transcribe", write_wav(tmp_path / "phone.wav", 48000, channels=2, seconds=0.5)])
    (audio,) = fake_engine
    assert audio.ndim == 1 and len(audio) == 8000 and audio.dtype == np.float32
    peak = np.abs(np.fft.rfft(audio)).argmax() * 16000 / len(audio)
    assert abs(peak - 440) < 5 and 0.25 < np.abs(audio).max() < 0.35  # same tone, same loudness
    assert capsys.readouterr().out == "ሞዴል ይሰራል\n"


def test_transcribe_a_missing_file_says_so(tmp_path):
    missing = str(tmp_path / "missing.wav")
    assert exit_message(["transcribe", missing]) == f"qal-scribe: no such file: {missing}"


def test_transcribe_says_on_stderr_when_it_downloads_first(tmp_path, capsys, monkeypatch, fake_engine, hf_cache):
    downloads = []
    monkeypatch.setattr(models, "download", lambda spec, with_lm=False: downloads.append(spec.language))
    monkeypatch.setattr("qal_scribe.translate.download", lambda model: downloads.append("nllb"))
    monkeypatch.setattr("qal_scribe.translate.Translator.load", lambda self: None)
    asked = []
    monkeypatch.setattr("qal_scribe.translate.Translator.translate",
                        lambda self, text, language: asked.append((text, language)) or "it works")
    cli.main(["transcribe", "--translate", write_wav(tmp_path / "a.wav", 16000)])
    captured = capsys.readouterr()
    assert downloads == ["nllb"] and captured.out == "it works\n" and asked == [(["ሞዴል", "ይሰራል"], "amh")]
    assert "fetching the translation model facebook/nllb-200-distilled-600M" in captured.err


@pytest.mark.parametrize(("platform", "tool"), [("darwin", "afconvert -f WAVE -d LEI16@16000 -c 1 "),
                                                ("linux", "ffmpeg -i ")])
def test_transcribe_an_unreadable_file_suggests_a_converter_that_is_there(tmp_path, monkeypatch, platform, tool):
    memo = tmp_path / "New Recording 3.m4a"
    memo.write_bytes(b"\x00\x00\x00\x20ftypM4A " + bytes(64))  # what Voice Memos saves; libsndfile cannot read it
    monkeypatch.setattr(sys, "platform", platform)
    message = exit_message(["transcribe", str(memo)])
    assert f"Convert it with: {tool}'{memo}'" in message
