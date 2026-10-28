"""CTC decoding: greedy today; language-model beam search is workstream `lm`.

Contract: decode(logits, processor, spec, lm=None) -> str. `logits` is a [frames, vocab] float32 numpy array.
"""
import numpy as np

from qal_scribe.models import ModelSpec


def greedy(logits: np.ndarray, processor, spec: ModelSpec) -> str:
    text = processor.batch_decode(logits.argmax(-1)[None])[0]
    if spec.output_tag:
        text = text.replace(spec.output_tag, "")
    return " ".join(text.split())


def load_lm(model_dir, spec: ModelSpec):
    """Return an object with .decode(logits) -> str using spec.lm_file inside model_dir, or None when the LM
    or its decoder library is unavailable (callers then fall back to greedy). Workstream `lm`."""
    return


def decode(logits: np.ndarray, processor, spec: ModelSpec, lm=None) -> str:
    return greedy(logits, processor, spec) if lm is None else lm.decode(logits)
