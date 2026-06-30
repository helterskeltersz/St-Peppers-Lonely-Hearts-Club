"""Window-aware deterministic step solver (matriks §4).

The competition window is NOT random — it is derived deterministically from the
dataset pair count (which is itself random in 10..50). This mirrors the verified
mechanic in rayonlabs/G.O.D `validator/tasks/synthetics/diffusion.py:594-607`
(see FASE0_VERIFY.md row g):

    clamped = clamp(num_pairs, 10, 50)
    scale   = (clamped - 10) / (50 - 10)
    hours   = 0.5 + scale * (1.0 - 0.5)
    hours   = ceil(hours * 4) / 4          # quantise to 15-min steps
    # qwen-image gets +0.5h

So once the zip is opened we already know the window. Throughput is then measured
live on the first N steps to convert window -> target_steps. No recipe numbers here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Verified constants (validator/tasks/synthetics/constants.py:61-72 @ G.O.D).
MIN_IMAGE_SYNTH_PAIRS = 10
MAX_IMAGE_SYNTH_PAIRS = 50
MIN_IMAGE_COMPETITION_HOURS = 0.5
MAX_IMAGE_COMPETITION_HOURS = 1.0
QWEN_IMAGE_EXTRA_COMPETITION_HOURS = 0.5

CHECKPOINT_EVERY_N_STEPS = 250
DEFAULT_TIMING_STEPS = 50          # measure it/s over the first 50 steps
DEFAULT_SAFETY = 0.82              # matriks §4: 0.80-0.85 of the window
DEFAULT_UPLOAD_RESERVE_FRAC = 0.10 # keep ~10% of the window to save + upload


def predict_window_hours(num_pairs: int, model_type: str) -> float:
    """Deterministic window length in hours for a given dataset size + arch."""
    if MAX_IMAGE_SYNTH_PAIRS <= MIN_IMAGE_SYNTH_PAIRS:
        hours = MIN_IMAGE_COMPETITION_HOURS
    else:
        clamped = min(max(num_pairs, MIN_IMAGE_SYNTH_PAIRS), MAX_IMAGE_SYNTH_PAIRS)
        scale = (clamped - MIN_IMAGE_SYNTH_PAIRS) / (MAX_IMAGE_SYNTH_PAIRS - MIN_IMAGE_SYNTH_PAIRS)
        hours = MIN_IMAGE_COMPETITION_HOURS + scale * (MAX_IMAGE_COMPETITION_HOURS - MIN_IMAGE_COMPETITION_HOURS)
        hours = math.ceil(hours * 4) / 4.0
    if model_type == "qwen-image":
        hours = round(hours + QWEN_IMAGE_EXTRA_COMPETITION_HOURS, 2)
    return hours


@dataclass
class StepPlan:
    window_hours: float
    hours_to_complete: float      # authoritative value from the validator CLI
    it_per_s: float | None        # measured live; None until measured
    target_steps: int | None
    checkpoint_every: int = CHECKPOINT_EVERY_N_STEPS


def solve_target_steps(
    it_per_s: float,
    hours_to_complete: float,
    safety: float = DEFAULT_SAFETY,
    upload_reserve_frac: float = DEFAULT_UPLOAD_RESERVE_FRAC,
) -> int:
    """target_steps = floor(it/s * usable_seconds * safety).

    usable_seconds reserves a fraction of the window for the final checkpoint save
    + HF upload so we never get killed mid-write.
    """
    usable_seconds = hours_to_complete * 3600.0 * (1.0 - upload_reserve_frac)
    return int(math.floor(it_per_s * usable_seconds * safety))


def build_plan(
    num_pairs: int,
    model_type: str,
    hours_to_complete: float,
    it_per_s: float | None = None,
) -> StepPlan:
    """Assemble a StepPlan. If it_per_s is known (measured), target_steps is filled;
    otherwise target_steps stays None and the caller must measure first."""
    window = predict_window_hours(num_pairs, model_type)
    target = solve_target_steps(it_per_s, hours_to_complete) if it_per_s else None
    return StepPlan(
        window_hours=window,
        hours_to_complete=hours_to_complete,
        it_per_s=it_per_s,
        target_steps=target,
    )


def measure_throughput(step_iter=None, n_steps: int = DEFAULT_TIMING_STEPS) -> float:
    """Live it/s over the first `n_steps` from a trainer step source.

    Structure is wired (Fase 2a-1): it consumes `n_steps` items from `step_iter`
    (the trainer's per-step callback/generator) and returns steps/sec. But it REFUSES
    to run without CUDA — there is no honest it/s to report on CPU, and fabricating one
    would corrupt the step solver. The live measurement happens in Fase 2a-2 on GPU.
    """
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if not has_cuda:
        raise RuntimeError(
            "measure_throughput needs CUDA + a live trainer step source (Fase 2a-2). "
            "Refusing to fabricate it/s on CPU."
        )
    if step_iter is None:
        raise RuntimeError("measure_throughput needs a step_iter (trainer step callback).")

    import time
    start = time.perf_counter()
    count = 0
    for _ in zip(range(n_steps), step_iter):
        count += 1
    elapsed = time.perf_counter() - start
    if count == 0 or elapsed <= 0:
        raise RuntimeError("measure_throughput measured no steps")
    return count / elapsed
