"""Scoring harness constants (matriks §0 / §4).

These mirror the validator's scoring mechanic so a checkpoint can be scored locally
before spending GPU on an A/B. Values below are VERIFIED in MATRIKS (which cites the
G.O.D validator source); the EXACT L2 reduction (MSE vs RMSE) and the absolute scale
are only CONFIRMED on GPU by reproducing a known A/B (Flux gs85 0.06490 vs gs1
0.07098). Do NOT treat CPU/dummy runs as calibrated.
"""
from __future__ import annotations

# Composite score = 0.25*text_guided_L2 + 0.75*no_text_L2
# VERIFIED (matriks §0, cites validator/scoring/tasks.py:293-296).
W_TEXT = 0.25
W_NOTEXT = 0.75

# img2img reconstruction denoise strength per architecture.
# VERIFIED (matriks §0.4, cites validator/evaluation/constants.py:19-24).
DENOISE_STRENGTH = {
    "sdxl": 0.9,
    "flux": 0.75,
    "z-image": 0.90,
    "qwen-image": 0.93,
}

# 10 seeds from a fixed master seed, 2 passes (prompt on / off) per eval image.
# VERIFIED (matriks §0.5).
MASTER_SEED = 42
N_SEEDS = 10

# Held-out fraction (matriks notes ~80/10 split); local proxy default.
DEFAULT_HOLDOUT_FRAC = 0.2

# Known A/B anchor used ONLY on GPU to check the harness is calibrated (not asserted
# on CPU/dummy). If a real Flux run reproduces roughly this ordering/scale, the
# harness metric form is trusted.
CALIBRATION_ANCHOR = {
    "flux_gs85_L2": 0.06490,
    "flux_gs1_L2": 0.07098,
    "note": "gs85 must score LOWER than gs1; confirm on GPU, never from dummy.",
}


def denoise_for(model_type: str) -> float:
    return DENOISE_STRENGTH.get((model_type or "").lower(), 0.9)
