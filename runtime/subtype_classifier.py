"""Subtype classifier — STRUCTURE ONLY (Fase 1).

Subtype is NOT in the task payload (verified Fase 0), so it must be inferred at
runtime from the dataset signature, then mapped to one of three buckets from the
matriks (§3b):

    BUCKET_AGGRESSIVE  -> aggressive-fit  (person/face/object/concept)
    BUCKET_PRIOR       -> prior-preserving (style/scene)
    BUCKET_LOGO        -> logo/text specialist

This module computes the *signature* and returns a bucket + confidence. It does
NOT contain any recipe/hyperparameters — that mapping is Fase 2 (see recipe.py).

What is implemented now (cheap, dependency-free):
  - image_count
  - caption_entropy (token-level Shannon entropy across .txt captions)
  - trigger_present
What is stubbed (heavier deps, Fase 2): subject_variance (CLIP/face-embedding),
ocr_area_ratio (OCR). They return None and are flagged so the combiner never
silently treats "unknown" as "zero".
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, asdict

# Bucket identifiers (recipe mapping lives in recipe.py — intentionally empty in Fase 1).
BUCKET_AGGRESSIVE = "aggressive-fit"
BUCKET_PRIOR = "prior-preserving"
BUCKET_LOGO = "logo-specialist"

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class DatasetSignature:
    image_count: int
    caption_entropy: float | None      # token Shannon entropy; None if no captions
    trigger_present: bool
    subject_variance: float | None     # TODO Fase 2: CLIP/face-embedding spread
    ocr_area_ratio: float | None       # TODO Fase 2: mean OCR text-area fraction

    def to_dict(self) -> dict:
        return asdict(self)


def _list_images(dataset_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(_IMAGE_EXTS):
                out.append(os.path.join(root, f))
    return out


def _caption_entropy(dataset_dir: str) -> float | None:
    tokens: Counter[str] = Counter()
    found = False
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            found = True
            try:
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for tok in fh.read().lower().replace(",", " ").split():
                        tokens[tok] += 1
            except OSError:
                continue
    if not found or not tokens:
        return None
    total = sum(tokens.values())
    return -sum((c / total) * math.log2(c / total) for c in tokens.values())


def compute_signature(dataset_dir: str, trigger_word: str | None) -> DatasetSignature:
    images = _list_images(dataset_dir)
    return DatasetSignature(
        image_count=len(images),
        caption_entropy=_caption_entropy(dataset_dir),
        trigger_present=bool(trigger_word and str(trigger_word).strip()),
        subject_variance=None,   # TODO Fase 2
        ocr_area_ratio=None,     # TODO Fase 2
    )


def classify(dataset_dir: str, trigger_word: str | None) -> tuple[str, float, DatasetSignature]:
    """Return (bucket, confidence, signature).

    PROVISIONAL heuristic — thresholds are placeholders to be calibrated in Fase 2.
    Confidence stays low (<=0.5) while the two heavy signals are stubbed, so the
    caller knows not to trust the bucket blindly yet.
    """
    sig = compute_signature(dataset_dir, trigger_word)

    # Ground-truth-ish prior from Fase 0: trigger_word absent => style-leaning
    # (the validator uses trigger presence as a style/person hint). This is a
    # provisional lean, NOT a final rule.
    if not sig.trigger_present:
        bucket = BUCKET_PRIOR
    else:
        bucket = BUCKET_AGGRESSIVE

    # ocr_area_ratio (Fase 2) is what should promote to BUCKET_LOGO; stubbed now.
    confidence = 0.4  # capped low until subject_variance + ocr_area_ratio land
    return bucket, confidence, sig
