"""Subtype classifier (Fase 2a-1 — 4 signals live).

Infers a training bucket from the dataset signature (subtype isn't in the payload):
    BUCKET_AGGRESSIVE  aggressive-fit   (person/face/object/concept — tight subject)
    BUCKET_PRIOR       prior-preserving (style/scene — diverse subjects)
    BUCKET_LOGO        logo-specialist  (high text-area -> caption carries visible text)

Signals:
  image_count       — dependency-free
  caption_entropy   — token Shannon entropy across .txt
  trigger_present   — trigger word given?
  subject_variance  — CLIP embedding spread if open_clip is importable, else a CPU
                      PIL+numpy proxy (color-hist + grayscale layout). Always a float.
  ocr_area_ratio    — mean text-bbox area fraction via pytesseract; None if OCR stack absent.

Recipe mapping lives in recipe.py. Thresholds here are 📐 starting points (calibrate in 2a-2).
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, asdict

BUCKET_AGGRESSIVE = "aggressive-fit"
BUCKET_PRIOR = "prior-preserving"
BUCKET_LOGO = "logo-specialist"

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 📐 calibration thresholds (sweep in Fase 2a-2 with real datasets).
OCR_LOGO_THRESHOLD = 0.08          # mean text-area fraction >= 8% -> logo
# subject_variance scale differs by source: CLIP cosine distances are larger than the
# CPU proxy's. Below threshold -> tight subject (aggressive); above -> diverse (prior).
SUBJECT_VAR_LOW_CLIP = 0.30
SUBJECT_VAR_LOW_PROXY = 0.04
_MAX_IMAGES_FOR_HEAVY = 60         # cap heavy signals for speed on big sets


def _subject_var_threshold(source: str | None) -> float:
    return SUBJECT_VAR_LOW_PROXY if source == "proxy" else SUBJECT_VAR_LOW_CLIP


@dataclass
class DatasetSignature:
    image_count: int
    caption_entropy: float | None
    trigger_present: bool
    subject_variance: float | None
    subject_variance_source: str | None   # "clip" | "proxy" | None
    ocr_area_ratio: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _list_images(dataset_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(_IMAGE_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


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


# ---------------- subject_variance: CLIP preferred, PIL/numpy proxy fallback ----------------
def _subject_variance(image_paths: list[str]) -> tuple[float | None, str | None]:
    paths = image_paths[:_MAX_IMAGES_FOR_HEAVY]
    if len(paths) < 2:
        return None, None
    feats = _clip_features(paths)
    source = "clip"
    if feats is None:
        feats = _proxy_features(paths)
        source = "proxy"
    if feats is None:
        return None, None
    return _mean_pairwise_cosine_distance(feats), source


def _clip_features(paths: list[str]):
    """CLIP image embeddings if open_clip + torch are importable; else None."""
    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception:
        return None
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        import numpy as np
        vecs = []
        with torch.no_grad():
            for p in paths:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                v = model.encode_image(img)[0]
                v = v / v.norm()
                vecs.append(v.cpu().numpy())
        return np.stack(vecs)
    except Exception:
        return None


def _proxy_features(paths: list[str]):
    """CPU proxy: per-image feature = normalized RGB histogram + 8x8 grayscale layout."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    feats = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
        hist = np.concatenate([
            np.histogram(arr[:, :, c], bins=8, range=(0, 1))[0] for c in range(3)
        ]).astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        gray = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float32).flatten() / 255.0
        feats.append(np.concatenate([hist, gray]))
    if len(feats) < 2:
        return None
    import numpy as np
    return np.stack(feats)


def _mean_pairwise_cosine_distance(feats) -> float:
    import numpy as np
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sims = f @ f.T
    n = f.shape[0]
    iu = np.triu_indices(n, k=1)
    dist = 1.0 - sims[iu]
    return float(np.clip(dist.mean(), 0.0, 1.0))


# ---------------- ocr_area_ratio: pytesseract ----------------
def _ocr_area_ratio(image_paths: list[str]) -> float | None:
    paths = image_paths[:_MAX_IMAGES_FOR_HEAVY]
    if not paths:
        return None
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return None
    ratios = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception:
            return None  # tesseract binary missing -> signal unavailable
        # Count ONLY word-level boxes (level==5) with real alphanumeric text and high
        # confidence. tesseract returns nested boxes (page/block/para/line/word); summing
        # all levels overcounts massively and false-positives on abstract images.
        area = 0
        levels = data.get("level", [])
        for i in range(len(levels)):
            if levels[i] != 5:
                continue
            try:
                if float(data["conf"][i]) < 60:
                    continue
            except (TypeError, ValueError):
                continue
            text = str(data["text"][i]).strip()
            if not any(ch.isalnum() for ch in text):
                continue
            area += int(data["width"][i]) * int(data["height"][i])
        ratios.append(min(1.0, area / float(w * h)))
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def compute_signature(dataset_dir: str, trigger_word: str | None) -> DatasetSignature:
    images = _list_images(dataset_dir)
    var, var_src = _subject_variance(images)
    return DatasetSignature(
        image_count=len(images),
        caption_entropy=_caption_entropy(dataset_dir),
        trigger_present=bool(trigger_word and str(trigger_word).strip()),
        subject_variance=var,
        subject_variance_source=var_src,
        ocr_area_ratio=_ocr_area_ratio(images),
    )


def classify(dataset_dir: str, trigger_word: str | None) -> tuple[str, float, DatasetSignature]:
    """Return (bucket, confidence, signature). Confidence reflects signal margin
    (no longer capped — all 4 signals participate)."""
    sig = compute_signature(dataset_dir, trigger_word)

    # 1. Logo wins if OCR shows substantial text area.
    if sig.ocr_area_ratio is not None and sig.ocr_area_ratio >= OCR_LOGO_THRESHOLD:
        margin = (sig.ocr_area_ratio - OCR_LOGO_THRESHOLD) / max(OCR_LOGO_THRESHOLD, 1e-6)
        return BUCKET_LOGO, round(min(0.95, 0.6 + 0.35 * margin), 3), sig

    # 2. Else use subject_variance (+ trigger as a tie-breaker).
    if sig.subject_variance is not None:
        thr = _subject_var_threshold(sig.subject_variance_source)
        if sig.subject_variance <= thr or sig.trigger_present:
            bucket = BUCKET_AGGRESSIVE
            margin = (thr - sig.subject_variance) / max(thr, 1e-6)
        else:
            bucket = BUCKET_PRIOR
            margin = (sig.subject_variance - thr) / max(1.0 - thr, 1e-6)
        conf = 0.55 + 0.4 * max(0.0, min(1.0, margin))
        # CLIP is more trustworthy than the proxy.
        if sig.subject_variance_source == "proxy":
            conf *= 0.85
        return bucket, round(min(0.95, conf), 3), sig

    # 3. Fallback: only cheap signals available -> lean on trigger presence.
    bucket = BUCKET_AGGRESSIVE if sig.trigger_present else BUCKET_PRIOR
    return bucket, 0.45, sig
