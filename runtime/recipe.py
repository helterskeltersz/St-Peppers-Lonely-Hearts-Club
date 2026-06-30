"""Recipe resolver — PLACEHOLDER ONLY (Fase 1).

Per the Fase 1 rules: NO concrete hyperparameters here. This module defines the
*shape* of a recipe and a resolver that returns sentinels, so the entrypoint can
assemble a config skeleton and detect that the recipe is not yet filled. The actual
numbers (matriks §3 + unconditional override §3a) land in Fase 2.

Each field defaults to None (= "not set in Fase 1"). `is_complete()` tells the
entrypoint whether real training may proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


# TODO Fase 2: fill from matriks §1 baseline + §3a unconditional override + §3b/3c bucket deltas.
@dataclass
class Recipe:
    rank: int | None = None                 # network_dim / linear
    alpha: int | None = None                # network_alpha / linear_alpha
    conv: int | None = None                 # conv_dim (Z-Image baseline 16; else off)
    conv_alpha: int | None = None
    learning_rate: float | None = None
    optimizer: str | None = None            # AdamW8Bit / Adafactor / adamw8bit
    scheduler: str | None = None
    caption_dropout_rate: float | None = None  # §3a lever #1 (0.30-0.50) — Fase 2
    min_snr_gamma: float | None = None      # SDXL only; flow-matching arches ignore it
    discrete_flow_shift: float | None = None  # Flux only; DO NOT set for Qwen (toolkit derives it)
    timestep_type: str | None = None        # qwen/z = "weighted"
    noise_scheduler: str | None = None      # qwen/z = "flowmatch"
    resolution: str | None = None
    train_batch_size: int | None = None
    # step/epoch count is decided by the step_solver, not hardcoded here.


def resolve_recipe(model_type: str, bucket: str) -> Recipe:
    """Return an EMPTY recipe (all None) in Fase 1.

    The (model_type, bucket) -> numbers mapping is Fase 2. We keep the signature
    stable so the entrypoint and tests can already exercise the wiring.
    """
    return Recipe()


def is_complete(recipe: Recipe) -> bool:
    """A recipe is usable only when the training-critical fields are set.

    In Fase 1 this is always False (resolve_recipe returns sentinels), which makes
    the entrypoint stop before launching a real (garbage) training run.
    """
    required = ("rank", "alpha", "learning_rate", "optimizer")
    return all(getattr(recipe, name) is not None for name in required)


def missing_fields(recipe: Recipe) -> list[str]:
    return [f.name for f in fields(recipe) if getattr(recipe, f.name) is None]
