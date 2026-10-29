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


@pytest.mark.parametrize(("line", "message"), [
    ('restore_delay = "0.5"', "restore_delay must be a number"),
    ("restore_delay = true", "restore_delay must be a number"),
    ('lm = "no"', "lm must be true or false"),
    ("sounds = 1", "sounds must be true or false"),
    ("words_file = 5", "words_file must be text in quotes"),
    ("device = 0", "device must be text in quotes"),
    ('translate_key = ["ctrl_r"]', "translate_key must be text in quotes"),
])
def test_wrong_type_is_an_error_naming_the_key(tmp_path, line, message):
    p = tmp_path / "config.toml"
    p.write_text(line + "\n")
    with pytest.raises(ValueError, match=message):
        config.load(p)


def test_numbers_may_be_whole(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("restore_delay = 1\n")
    assert config.load(p).restore_delay == 1
