"""Step budget for ai-toolkit arches (F7).

Z-Image / Qwen count raw optimiser steps (batch 1). A sublinear power law in the
image count holds a roughly constant number of passes and completes AT the measured
optimum instead of training past the overfit cliff. A person/product identity set
memorises fast, so the SUBJECT path gets a much tighter budget than the logo/style
band.

VERIFIED against winner `zayden_recipe.py:_zayden_aitk_steps` (:356-374).
The window/throughput solver (step_solver.py) still applies as an UPPER CAP so a
budget never overruns the competition wall-clock.
"""
from __future__ import annotations

_STEPS = {
    "z-image":    {"base": 1100, "n_ref": 30, "min": 800, "max": 1400, "p": 0.5},
    "qwen-image": {"base": 1000, "n_ref": 30, "min": 700, "max": 1400, "p": 0.5},
}
_STEPS_SUBJECT = {
    "z-image":    {"base": 230, "n_ref": 14, "min": 120, "max": 680, "p": 0.85},
    "qwen-image": {"base": 210, "n_ref": 14, "min": 110, "max": 640, "p": 0.85},
}

_SUBJECT_CATS = {"person", "product"}


def aitoolkit_steps(model_type: str, n_images: int, category: str | None = None) -> int:
    cat = (category or "").lower().strip()
    table = _STEPS_SUBJECT if cat in _SUBJECT_CATS else _STEPS
    spec = table.get((model_type or "").lower())
    if not spec:
        return 1000
    n = max(1, int(n_images or 0))
    steps = spec["base"] * (n / spec["n_ref"]) ** spec["p"]
    return int(max(spec["min"], min(spec["max"], round(steps))))
