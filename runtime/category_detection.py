"""Six-category detection (v3) — replaces the Fase-1 3-bucket subtype_classifier.

Category is NOT in the task payload, so it is inferred from the dataset caption
vocabulary. Six categories drive both the training shape and the caption mode:

    person, product          -> SUBJECT  (Prodigy, identity lock)
    style, logo, social, design -> STYLE-like (AdamW, prior-preserving)

Design (VERIFIED against winner `trainer/utils/category_detection.py`):
  - keyword scoring per category on WORD BOUNDARIES (so "ui" != "building"),
  - a distinct-marker count >= threshold promotes the category,
  - otherwise fall back to style (if an art-style is present) or person.
The old trigger-presence short-circuit is REMOVED — it mis-routed logo/social/
design (no trigger) into the style/person path and never reached LOGO.
"""
from __future__ import annotations

import os
import re

from runtime.style_detection import detect_styles_in_prompts

# Shape groupings (Sumbu 1).
SUBJECT_CATEGORIES = {"person", "product"}
STYLE_CATEGORIES = {"style", "logo", "social", "design"}

# High-precision markers chosen to fire on the validator's synthetic logo/social/
# design/product prompts but NOT on ordinary person/style captions.
_CATEGORY_KEYWORDS = {
    "logo": [
        "logo", "logos", "logo mark", "lettermark", "wordmark", "monogram",
        "logotype", "brandmark", "brand mark", "emblem", "brand identity",
        "brand personality", "logo design", "logo collection", "vector logo",
        "vector mark", "typography direction", "icon mark",
    ],
    "social": [
        "social media", "social post", "social media post", "social media content",
        "instagram post", "instagram story", "campaign", "carousel", "story format",
        "call to action", "cta", "promotional graphic", "ad creative", "feed post",
        "announcement post", "headline", "subheadline",
    ],
    "design": [
        "ui", "ux", "ui design", "ux design", "user interface", "app screen",
        "mobile app", "mobile screen", "web app", "landing page", "app interface",
        "web dashboard", "dashboard screen", "ui collection", "interface screenshot",
        "wireframe", "navigation bar", "nav bar", "tab bar", "onboarding screen",
        "ui screenshot", "ui kit",
    ],
    "product": [
        "product photography", "product shot", "single product", "consistent product",
        "product trigger", "product variant", "isolated product", "product reference",
        "product mockup", "product render", "colorway", "packaging shot", "studio product",
    ],
}
# All four task-shaped categories need a clear signal; product mis-detection safely
# falls back to person (also a Prodigy subject path).
_CATEGORY_MIN_HITS = 3

_KW_PATTERNS = {
    cat: [re.compile(rf"\b{re.escape(kw)}\b") for kw in kws]
    for cat, kws in _CATEGORY_KEYWORDS.items()
}


def read_caption_prompts(train_data_dir: str) -> list[str]:
    prompts: list[str] = []
    for root, _, files in os.walk(train_data_dir):
        for f in files:
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(root, f), encoding="utf-8") as fh:
                        prompts.append(fh.read())
                except OSError:
                    continue
    return prompts


def detect_category(prompts: list[str]) -> str:
    """Return one of person/style/logo/social/design/product.

    Category vocabulary is scored FIRST (a logo/UI task often also names a style),
    then an art-style short-circuit, then person as the base default.
    """
    joined = " \n ".join(p.lower() for p in prompts if p)
    if not joined.strip():
        return "person"
    scores = {cat: sum(1 for pat in pats if pat.search(joined))
              for cat, pats in _KW_PATTERNS.items()}
    best = max(scores, key=scores.get)
    if scores[best] >= _CATEGORY_MIN_HITS:
        return best
    # Style short-circuit: a real art-style term must appear in >=25% of captions
    # (winner threshold). A stray "aesthetic" in a person-triggered set no longer
    # flips the whole task to style.
    if detect_styles_in_prompts(prompts):
        return "style"
    return "person"


def category_shape(category: str) -> str:
    """'subject' (Prodigy) or 'style' (AdamW) for a category."""
    return "subject" if (category or "").lower() in SUBJECT_CATEGORIES else "style"


def is_style_like(category: str) -> bool:
    return (category or "").lower() in STYLE_CATEGORIES


def detect_trigger(prompts: list[str]) -> str | None:
    """Recover the trigger: the short leading comma-tag (<=4 words) repeated in a
    clear majority (>=60%) of captions. Used when the validator omits --trigger-word.
    """
    counts: dict[str, int] = {}
    originals: dict[str, str] = {}
    n = 0
    for p in prompts:
        if not p or not p.strip():
            continue
        n += 1
        lead = p.split(",")[0].strip()
        norm = lead.lower()
        if not norm or len(norm.split()) > 4:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        originals.setdefault(norm, lead)
    if n == 0 or not counts:
        return None
    best_norm, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count >= max(2, int(0.6 * n)):
        return originals[best_norm]
    return None
