"""DEPRECATED (v3) — superseded by `runtime/recipe_engine.py`.

Fase-1 `recipe.py` was an empty placeholder whose comments still carried the killed
v2 thesis (unconditional push, caption_dropout 0.30-0.50, UNet-only). v3 removes that
thesis entirely (see MATRIKS_v3 changelog) and resolves recipes from the 3-axis
cascade in `recipe_engine.py`.

Kept only as a thin re-export so any external import of `runtime.recipe` keeps
working. New code should import `runtime.recipe_engine` directly.
"""
from __future__ import annotations

from runtime.recipe_engine import (  # noqa: F401
    resolve_recipe,
    build_aitoolkit_config,
    size_bucket,
    qwen_capacity,
    sanity_ok,
)
