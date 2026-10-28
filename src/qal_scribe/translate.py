"""Amharic -> English with facebook/nllb-200-distilled-600M (workstream `translate`). Loaded lazily on first use.

Contract: t = Translator(device); t.load(); t.translate(amharic_text) -> english_text
"""


class Translator:
    MODEL = "facebook/nllb-200-distilled-600M"
    REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def load(self) -> None:
        raise NotImplementedError

    def translate(self, text: str) -> str:
        raise NotImplementedError
