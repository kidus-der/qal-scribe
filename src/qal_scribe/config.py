"""User config from ~/.config/qal-scribe/config.toml (or $XDG_CONFIG_HOME). Missing file -> defaults.

Contract: load(path=None) -> Config (ValueError for unknown keys or wrong types); load_words(cfg) -> the words.txt
replacements, {heard: wanted}.
"""
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "qal-scribe"


@dataclass
class Config:
    language: str = "amh"
    dictate_key: str = "alt_r"  # pynput Key name
    translate_key: str | None = "ctrl_r"  # None disables the translate hotkey
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
    pairs = {}
    for line in cfg.words_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=>" in line:
            heard, wanted = (s.strip() for s in line.split("=>", 1))
            if heard:
                pairs[heard] = wanted
    return pairs


def load(path: Path | None = None) -> Config:
    path = path or config_dir() / "config.toml"
    if not path.exists():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")
    check_types(data, path)
    if "words_file" in data:
        data["words_file"] = Path(data["words_file"]).expanduser()
    return Config(**data)
