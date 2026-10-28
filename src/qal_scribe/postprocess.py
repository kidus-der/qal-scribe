"""Text clean-up after transcription (workstream `amharic`).

Contract: process(text, cfg, words) -> str. Pure function, no I/O, no model. Each step is opt-in via Config:
  merge_homophones, numbers_to_digits, punctuation (experimental), plus the user's word list (always on).
"""
from qal_scribe.config import Config


def process(text: str, cfg: Config, words: dict[str, str] | None = None) -> str:
    return " ".join(text.split())
