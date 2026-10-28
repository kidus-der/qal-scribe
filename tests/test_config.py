import pytest

from qal_scribe import config


def test_missing_file_gives_defaults(tmp_path):
    cfg = config.load(tmp_path / "nope.toml")
    assert cfg.language == "amh" and cfg.dictate_key == "alt_r"


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('langauge = "amh"\n')
    with pytest.raises(ValueError, match="langauge"):
        config.load(p)


def test_words_file_expands_user(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('words_file = "~/w.txt"\nlm = false\n')
    cfg = config.load(p)
    assert not cfg.lm and "~" not in str(cfg.words_file)
