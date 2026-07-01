"""Pure scoring math (CPU-testable, no model / no GPU).

The composite reconstruction score:

    score = W_TEXT * mean(text_L2) + W_NOTEXT * mean(notext_L2)

where each *_L2 is a per-image pixel distance between the held-out original and its
img2img reconstruction (prompt-guided vs empty-prompt). These functions are exact
and deterministic; they carry NO calibration claim — the reduction form and scale
are confirmed on GPU against a known A/B (see constants.CALIBRATION_ANCHOR).
"""
from __future__ import annotations

from typing import Iterable, Sequence

from harness.constants import W_TEXT, W_NOTEXT


def to_array(img):
    """Normalise a PIL image / path / array to float32 [0,1] HxWxC. Import kept local
    so the pure math below can be imported without Pillow present."""
    import numpy as np
    if isinstance(img, str):
        from PIL import Image
        img = Image.open(img).convert("RGB")
    if hasattr(img, "convert"):  # PIL image
        import numpy as np
        img = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    else:
        img = np.asarray(img, dtype="float32")
        if img.max() > 1.0:
            img = img / 255.0
    return img


def pixel_l2(a, b, reduction: str = "mse") -> float:
    """Per-image pixel distance. reduction: 'mse' (default), 'rmse', or 'mae'.

    The exact reduction the validator uses is confirmed on GPU; 'mse' is the default
    working assumption. Both inputs must be the same shape, float [0,1]."""
    import numpy as np
    a = to_array(a)
    b = to_array(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    if reduction == "mae":
        return float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    if reduction == "rmse":
        return float(mse ** 0.5)
    if reduction == "mse":
        return mse
    raise ValueError(f"unknown reduction {reduction!r}")


def weighted_score(text_l2: float, notext_l2: float,
                   w_text: float = W_TEXT, w_notext: float = W_NOTEXT) -> float:
    """Composite: 0.25*text + 0.75*notext. Lower is better."""
    return w_text * float(text_l2) + w_notext * float(notext_l2)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(text_l2s: Sequence[float], notext_l2s: Sequence[float]) -> dict:
    """Aggregate per-(image,seed) L2s into the composite score + components."""
    t = mean(text_l2s)
    nt = mean(notext_l2s)
    return {
        "score": weighted_score(t, nt),
        "text_l2": t,
        "notext_l2": nt,
        "n_text": len(list(text_l2s)),
        "n_notext": len(list(notext_l2s)),
    }
