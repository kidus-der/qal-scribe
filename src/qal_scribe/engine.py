"""Loads a speech model once and turns 16 kHz mono float32 audio into text.

Contract (used by Controller and `qal-scribe transcribe`):
    engine = Engine(spec, device=None, use_lm=True); engine.load(on_progress)
    result = engine.transcribe(audio)  -> Result(text, compute_seconds)
"""
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qal_scribe import decode
from qal_scribe.models import ModelSpec, download

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Result:
    text: str
    compute_seconds: float


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Engine:
    def __init__(self, spec: ModelSpec, device: str | None = None, use_lm: bool = True):
        self.spec = spec
        self.requested_device = device
        self.use_lm = use_lm
        self.device = None
        self.lm = None

    def load(self, on_progress: Callable[[str], None] = lambda _: None) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        on_progress(f"fetching {self.spec.repo}")
        path = download(self.spec, with_lm=self.use_lm)
        self.device = pick_device(self.requested_device)
        # fp16 on GPU: identical text to fp32 on hohe-asr, ~2x faster on MPS (Amharic/hohe-asr-amharic/bench.py).
        self.dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        on_progress(f"loading on {self.device}")
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForCTC.from_pretrained(path, dtype=self.dtype).eval().to(self.device)
        if self.use_lm:
            on_progress("loading language model")
            self.lm = decode.load_lm(path, self.spec)
        self.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"))  # warm-up: GPU kernels compile on first use

    def logits(self, audio: np.ndarray) -> np.ndarray:
        import torch

        x = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device) for k, v in x.items()}
        with torch.inference_mode():
            return self.model(**x).logits[0].float().cpu().numpy()

    def transcribe(self, audio: np.ndarray) -> Result:
        t = time.perf_counter()
        text = decode.decode(self.logits(audio), self.processor, self.spec, self.lm)
        return Result(text, time.perf_counter() - t)
