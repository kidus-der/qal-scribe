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
    ('indicator = "yes"', "indicator must be true or false"),
    ("words_file = 5", "words_file must be text in quotes"),
    ("device = 0", "device must be text in quotes"),
    ('translate_key = ["ctrl_r"]', "translate_key must be text in quotes"),
    ('translation = "no"', "translation must be true or false"),
    ("translation_model = 3", "translation_model must be text in quotes"),
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


def test_words_file_with_a_bom_keeps_its_first_entry(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("ኤፒአይ => API\n", encoding="utf-8-sig")  # what Notepad's "UTF-8" writes
    assert config.load_words(config.Config(words_file=p)) == {"ኤፒአይ": "API"}


def test_words_file_that_is_not_utf8_or_not_a_file_names_itself(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("ኤፒአይ => API\n", encoding="utf-16")
    with pytest.raises(ValueError, match=r"words\.txt: not UTF-8 text"):
        config.load_words(config.Config(words_file=p))
    with pytest.raises(ValueError, match=str(tmp_path)):
        config.load_words(config.Config(words_file=tmp_path))  # a folder


@pytest.mark.parametrize(("platform", "key"), [("darwin", "cmd_r"), ("linux", "ctrl_r")])
def test_translate_key_default_exists_on_the_platform_keyboard(monkeypatch, platform, key):
    monkeypatch.setattr("sys.platform", platform)
    assert config.Config().translate_key == key


def test_save_keeps_comments_and_changes_the_line_in_place(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('# my note\nsounds = true  # old\n\n# theme = "qal"\nlm = false\n')
    cfg = config.save(p, {"sounds": False, "theme": "nord"})
    assert not cfg.sounds and cfg.theme == "nord" and not cfg.lm
    assert p.read_text() == '# my note\nsounds = false\n\ntheme = "nord"\nlm = false\n'
    assert config.load(p) == cfg


def test_save_uncomments_the_template_line_where_it_is(tmp_path):
    p = tmp_path / "config.toml"
    config.save(p, {"mic": "Blue Yeti"})  # missing file: starts from the commented template
    lines = p.read_text().splitlines()
    assert 'mic = "Blue Yeti"' in lines and not any(line.startswith("# mic =") for line in lines)
    assert sum(line.startswith("#") for line in lines) > 20  # the explanations survive
    assert config.load(p).mic == "Blue Yeti"


def test_save_none_removes_the_line_and_appends_new_keys(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('mic = "x"\n')
    config.save(p, {"mic": None, "preview": False})
    assert p.read_text() == "preview = false\n"


def test_save_rejects_bad_values_and_leaves_the_file_alone(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("sounds = true\n")
    for changes in ({"translation_model": "nope"}, {"language": "xx"}, {"nonsense": 1}, {"dictate_key": ""},
                    {"restore_delay": "slow"}):
        with pytest.raises(ValueError):
            config.save(p, changes)
    assert p.read_bytes() == b"sounds = true\n" and not list(tmp_path.glob("*.tmp"))


def test_save_writes_text_with_quotes_as_valid_toml(tmp_path):
    p = tmp_path / "config.toml"
    assert config.save(p, {"mic": 'Mic "A" \\ 2'}).mic == 'Mic "A" \\ 2'
