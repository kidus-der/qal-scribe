"""Loads a speech model once and turns 16 kHz mono float32 audio into text.

Contract (used by Controller and `qal-scribe transcribe`):
    engine = Engine(spec, device=None, use_lm=True); engine.load(on_progress)
    result = engine.transcribe(audio)  -> Result(text, compute_seconds); any length (split at quiet points)
    result = engine.preview(audio)     -> Result; greedy draft (no LM) of the last 30 s
    engine.unload()                    frees the weights and GPU memory
Not thread-safe: the caller serialises calls (Controller holds one lock around every model call).
"""
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qal_scribe import decode, models
from qal_scribe.models import ModelSpec, download

SAMPLE_RATE = 16000
MAX_CHUNK = 30 * SAMPLE_RATE  # the model is trained on clips up to ~30 s
PAUSE = SAMPLE_RATE // 5  # a pause must be this long to beat a gap between words
MIN_SEARCH = 3 * SAMPLE_RATE  # widest range a cut may be forced into


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


def quietest(audio: np.ndarray) -> int:
    """Index of the middle of the quietest PAUSE-long stretch of `audio`."""
    w = min(PAUSE, len(audio))
    cum = np.concatenate(([0.0], np.cumsum(audio.astype("float64") ** 2)))
    return int(np.argmin(cum[w:] - cum[:-w])) + w // 2


def split_quiet(audio: np.ndarray, max_len: int = MAX_CHUNK) -> list[np.ndarray]:
    """Cut `audio` into chunks of at most max_len samples at the quietest points.

    Chunks stay about even (each cut is searched within 70-130% of an even split), so none is a stub.
    Aims for the fewest chunks, but never squeezes the search below MIN_SEARCH: a quiet cut beats saving a chunk.
    """
    chunks = []
    while len(audio) > max_len:
        n = -(-len(audio) // max_len)  # chunks still needed
        target = len(audio) // n
        hi = min(max_len, int(target * 1.3))
        lo = min(max(int(target * 0.7), len(audio) - (n - 1) * max_len), hi - MIN_SEARCH)  # rest fits in n-1
        cut = lo + quietest(audio[lo:hi])
        chunks.append(audio[:cut])
        audio = audio[cut:]
    return [*chunks, audio]


class Engine:
    def __init__(self, spec: ModelSpec, device: str | None = None, use_lm: bool = True):
        self.spec = spec
        self.requested_device = device
        self.use_lm = use_lm
        self.device = None
        self.lm = None
        self.model = self.processor = None

    def load(self, on_progress: Callable[[str], None] = lambda _: None) -> None:
        import torch

        on_progress(f"fetching {self.spec.repo}")
        path = download(self.spec, with_lm=self.use_lm)
        self.device = pick_device(self.requested_device)
        # fp16 on GPU: identical text to fp32 on hohe-asr, ~2x faster on MPS (Amharic/hohe-asr-amharic/bench.py).
        self.dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        on_progress(f"loading on {self.device}")
        self.processor, model = self._load_weights(path, on_progress)
        self.model = model.eval().to(self.device)
        if self.use_lm:
            on_progress("loading language model")
            self.lm = decode.load_lm(path, self.spec)
        self.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"))  # warm-up: GPU kernels compile on first use

    def _load_weights(self, path, on_progress: Callable[[str], None]):
        """(processor, model) from the fp16 copy in the user cache; the first GPU run builds that copy
        from the Hugging Face snapshot (half the bytes and no fp32->fp16 cast on later starts)."""
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        gpu = self.dtype == torch.float16
        copy = models.fp16_copy(self.spec) if gpu else None
        if copy:
            try:
                return AutoProcessor.from_pretrained(copy), AutoModelForCTC.from_pretrained(copy, dtype=self.dtype)
            except Exception as e:  # noqa: BLE001 - a damaged cache must never stop start-up; rebuild it instead
                on_progress(f"model cache unusable ({type(e).__name__}); rebuilding")
                models.drop_fp16(self.spec)
        processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForCTC.from_pretrained(path, dtype=self.dtype)
        if gpu:
            on_progress("saving fp16 copy for faster starts")
            try:
                models.save_fp16(self.spec, lambda d: (model.save_pretrained(d), processor.save_pretrained(d)))
            except OSError as e:
                on_progress(f"could not write model cache: {e}")
        return processor, model

    def unload(self) -> None:
        """Free the weights and give GPU memory back."""
        import torch

        self.model = self.processor = self.lm = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def logits(self, audio: np.ndarray) -> np.ndarray:
        import torch

        x = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device) for k, v in x.items()}
        with torch.inference_mode():
            return self.model(**x).logits[0].float().cpu().numpy()

    def transcribe(self, audio: np.ndarray) -> Result:
        start = time.perf_counter()
        texts = [decode.decode(self.logits(c), self.processor, self.spec, self.lm) for c in split_quiet(audio)]
        return Result(" ".join(t for t in texts if t), time.perf_counter() - start)

    def preview(self, audio: np.ndarray) -> Result:
        """Greedy draft of the last 30 s for live feedback; cheaper than transcribe (no LM, one chunk)."""
        start = time.perf_counter()
        text = decode.greedy(self.logits(audio[-MAX_CHUNK:]), self.processor, self.spec)
        return Result(text, time.perf_counter() - start)
