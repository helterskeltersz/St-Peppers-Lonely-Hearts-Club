"""Per-category noise config (F5).

Flat-graphic and identity categories use multi-resolution pyramid noise instead of
a flat brightness offset: a constant offset smears hard edges / low-frequency
identity structure, while pyramid noise preserves high-frequency detail. Pyramid
noise and noise_offset are mutually exclusive, so the offset is zeroed when
multires is on.

VERIFIED against winner `zayden_recipe.py:_zayden_category_noise` (:233-252):
  person, logo, social, design -> multires (iterations=6, discount=0.3, offset=0)
  product, style, unknown      -> {} (inherit the recipe/bucket noise_offset)
"""
from __future__ import annotations

MULTIRES_ITERATIONS = 6
MULTIRES_DISCOUNT = 0.3

_MULTIRES_CATEGORIES = {"person", "logo", "social", "design"}


def category_noise(category: str | None) -> dict:
    cat = (category or "").lower().strip()
    if cat in _MULTIRES_CATEGORIES:
        return {
            "noise_offset": 0.0,
            "noise_offset_type": "Original",
            "multires_noise_iterations": MULTIRES_ITERATIONS,
            "multires_noise_discount": MULTIRES_DISCOUNT,
        }
    return {}
