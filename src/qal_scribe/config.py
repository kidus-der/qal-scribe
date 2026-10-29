"""User config from ~/.config/qal-scribe/config.toml (or $XDG_CONFIG_HOME). Missing file -> defaults."""
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
    preview: bool = True  # show a live draft while recording


def load(path: Path | None = None) -> Config:
    path = path or config_dir() / "config.toml"
    if not path.exists():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")
    if "words_file" in data:
        data["words_file"] = Path(data["words_file"]).expanduser()
    return Config(**data)
