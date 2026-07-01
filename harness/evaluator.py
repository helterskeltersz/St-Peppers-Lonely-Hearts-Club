"""Evaluation orchestration (structure CPU-testable; real generation is GPU).

`evaluate_checkpoint` runs the validator-shaped loop: for each held-out image, for
N seeds, reconstruct it via img2img at the arch's denoise strength, once with the
prompt (text) and once with an empty prompt (notext), then score with the pure math
in `score.py`.

The actual diffusion reconstruction is injected as `reconstruct_fn` so the
orchestration is testable on CPU with a dummy fn. At the GPU phase `reconstruct_fn`
wraps the real pipeline (LoRA loaded, denoise=DENOISE_STRENGTH[arch]). NO calibration
is asserted here; a run is only trusted once it reproduces CALIBRATION_ANCHOR on GPU.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from harness import constants as C
from harness.score import pixel_l2, aggregate

# reconstruct_fn(original, prompt, seed, denoise, model_type) -> reconstructed image
ReconstructFn = Callable[[object, Optional[str], int, float, str], object]


@dataclass
class EvalItem:
    image: object          # original held-out image (path / PIL / array)
    prompt: str            # source prompt for the text-guided pass


def holdout_split(items: List, frac: float = C.DEFAULT_HOLDOUT_FRAC,
                  seed: int = C.MASTER_SEED) -> tuple[list, list]:
    """Deterministic train/held-out split. Held-out is at least 1 item when possible."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_hold = max(1, round(len(items) * frac)) if items else 0
    hold = {idx[i] for i in range(n_hold)}
    held = [items[i] for i in range(len(items)) if i in hold]
    train = [items[i] for i in range(len(items)) if i not in hold]
    return train, held


def seeds_for(master: int = C.MASTER_SEED, n: int = C.N_SEEDS) -> list[int]:
    return [master + i for i in range(n)]


def evaluate_checkpoint(
    eval_items: List[EvalItem],
    model_type: str,
    reconstruct_fn: ReconstructFn,
    n_seeds: int = C.N_SEEDS,
    reduction: str = "mse",
) -> dict:
    """Score a checkpoint. Two passes per (image, seed): prompt-guided + empty-prompt.

    Returns aggregate() dict plus per-item breakdown. Pure orchestration — the model
    lives entirely inside reconstruct_fn."""
    denoise = C.denoise_for(model_type)
    seeds = seeds_for(n=n_seeds)
    text_l2s: List[float] = []
    notext_l2s: List[float] = []
    per_item = []
    for item in eval_items:
        it_text, it_notext = [], []
        for s in seeds:
            recon_text = reconstruct_fn(item.image, item.prompt, s, denoise, model_type)
            recon_notext = reconstruct_fn(item.image, None, s, denoise, model_type)
            it_text.append(pixel_l2(item.image, recon_text, reduction))
            it_notext.append(pixel_l2(item.image, recon_notext, reduction))
        text_l2s += it_text
        notext_l2s += it_notext
        per_item.append({"text_l2": sum(it_text) / len(it_text),
                         "notext_l2": sum(it_notext) / len(it_notext)})
    result = aggregate(text_l2s, notext_l2s)
    result.update({"model_type": model_type, "denoise": denoise,
                   "n_items": len(eval_items), "n_seeds": n_seeds,
                   "reduction": reduction, "per_item": per_item,
                   "calibrated": False})  # only set True after GPU anchor check
    return result


def compare_runs(run_a: dict, run_b: dict) -> dict:
    """Compare two scored runs (lower score wins). Used for edge A/B decisions."""
    delta = run_b["score"] - run_a["score"]
    winner = "a" if run_a["score"] < run_b["score"] else "b"
    return {"winner": winner, "score_a": run_a["score"], "score_b": run_b["score"],
            "delta": delta,
            "pct_change": (delta / run_a["score"] * 100) if run_a["score"] else 0.0}
