"""User config from ~/.config/qal-scribe/config.toml (or $XDG_CONFIG_HOME). Missing file -> defaults.

Contract: translate_key defaults to cmd_r on macOS, ctrl_r elsewhere. parse(text, path) / load(path=None) -> Config
(ValueError for unknown keys or wrong types); save(path, {key: value}) edits those lines in place, keeping every
comment, checks the result like load() plus doctor.config_problem, and replaces the file atomically (ValueError
leaves it untouched; a None value removes the line); load_words(cfg) -> the words.txt replacements, {heard: wanted}
(ValueError naming the file when it cannot be read as UTF-8 text).
"""
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "qal-scribe"


def default_translate_key() -> str:
    return "cmd_r" if sys.platform == "darwin" else "ctrl_r"  # Apple keyboards have no Right Control


@dataclass
class Config:
    language: str = "amh"
    dictate_key: str = "alt_r"  # pynput Key name
    translate_key: str | None = field(default_factory=default_translate_key)  # None disables the translate hotkey
    translation: bool = True  # False: no translate hotkey, and the translation model is never fetched or loaded
    translation_model: str = "nllb-600m"  # see translate.TRANSLATION_MODELS
    device: str | None = None  # None = auto (mps > cuda > cpu)
    lm: bool = True  # use the model's language model when available
    numbers_to_digits: bool = False
    merge_homophones: bool = False
    punctuation: bool = False  # experimental, rule-based
    words_file: Path = field(default_factory=lambda: config_dir() / "words.txt")
    restore_delay: float = 0.5
    sounds: bool = True
    theme: str = "qal"
    compact: bool = False
    mic: str | None = None  # input device name (substring) or index; None = system default
    preview: bool = True  # show a live draft while recording
    indicator: bool = True  # macOS: floating "Listening" pill while the terminal is hidden; ignored elsewhere
    check_updates: bool = True  # ask PyPI once a day whether a newer qal-scribe is out


# What TOML must hold for each field type; TOML has no null, so "str | None" fields are strings too.
KINDS = {bool: ("true or false", bool), float: ("a number", (int, float)), Path: ("text in quotes", str)}


def check_types(data: dict, path: Path) -> None:
    """Raise ValueError for a value of the wrong type: `restore_delay = "0.5"` would otherwise pass every
    check and then crash at paste time, after the clipboard was overwritten."""
    for f in fields(Config):
        if f.name not in data:
            continue
        value = data[f.name]
        what, types = KINDS.get(f.type, ("text in quotes", str))
        if not isinstance(value, types) or (f.type is not bool and isinstance(value, bool)):
            raise ValueError(f"{path}: {f.name} must be {what}, got {value!r}")


def load_words(cfg: Config) -> dict[str, str]:
    """words.txt lines: `heard => wanted` replacements; blank lines and # comments ignored."""
    if not cfg.words_file.exists():
        return {}
    try:
        text = cfg.words_file.read_text(encoding="utf-8-sig")  # -sig: Notepad's BOM would hide the first entry
    except UnicodeDecodeError as e:
        raise ValueError(f"{cfg.words_file}: not UTF-8 text ({e.reason} at byte {e.start}); save it as UTF-8") from e
    except OSError as e:  # e.g. words_file is a folder
        raise ValueError(f"{cfg.words_file}: {e.strerror or e}") from e
    pairs = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=>" in line:
            heard, wanted = (s.strip() for s in line.split("=>", 1))
            if heard:
                pairs[heard] = wanted
    return pairs


def load(path: Path | None = None) -> Config:
    path = path or config_dir() / "config.toml"
    return parse(path.read_text(encoding="utf-8"), path) if path.exists() else Config()


def parse(text: str, path: Path) -> Config:
    data = tomllib.loads(text)
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")
    check_types(data, path)
    if "words_file" in data:
        data["words_file"] = Path(data["words_file"]).expanduser()
    return Config(**data)


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value) if isinstance(value, str) else repr(value)  # a JSON string is a TOML basic string


def edit(text: str, key: str, value: object | None) -> str:
    """`text` with the `key = value` line replaced where it is, else where the template comments it out, else
    appended; value None removes an active line."""
    lines = text.splitlines()
    active = next((i for i, line in enumerate(lines) if re.match(rf"\s*{key}\s*=", line)), None)
    if value is None:
        if active is not None:
            del lines[active]
    else:
        new = f"{key} = {toml_value(value)}"
        template = next((i for i, line in enumerate(lines) if re.match(rf"#\s*{key}\s*=", line)), None)
        if active is not None or template is not None:
            lines[template if active is None else active] = new
        else:
            lines.append(new)
    return "\n".join(lines) + "\n" if lines else ""


def save(path: Path, changes: dict[str, object | None]) -> Config:
    from importlib.resources import files

    from qal_scribe.doctor import config_problem

    known = {f.name for f in fields(Config)}
    if unknown := set(changes) - known:
        raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")
    text = (path.read_text(encoding="utf-8") if path.exists()
            else files("qal_scribe").joinpath("default_config.toml").read_text(encoding="utf-8"))
    for key, value in changes.items():
        text = edit(text, key, value)
    cfg = parse(text, path)
    if problem := config_problem(cfg):
        raise ValueError(problem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return cfg
