"""Perceptual near-duplicate removal, pre-caption (F4).

Synthetic generators emit near-identical variants; under a fixed folder repeat a
duplicate gets full Nx gradient weight and the LoRA memorises that one frame — the
overfitting the pixel-L2 metric punishes. We drop redundant copies BEFORE the
(expensive) caption pass to trade re-exposure for diverse coverage.

Design (VERIFIED against winner `scripts/core/zayden_dedup.py`):
  - dHash 64-bit (structure) + mean-RGB colour signature; BOTH must match, so a
    same-shape/different-colour flat graphic (two brand logos) is NOT merged,
  - union-find grouping at a conservative Hamming threshold,
  - keep-floor so small sets are never thinned below a safe size.
Pure PIL + numpy — no extra dependency (trainer image already ships both).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SIDECAR_EXTS = {".txt", ".npz", ".npy", ".caption"}

# VERIFIED thresholds (zayden_dedup.dedup_dataset defaults).
MAX_HAMMING = 6
COLOUR_THRESH = 14.0
MIN_N = 15
MIN_KEEP_FRAC = 0.6


def _signature(path: str, hash_size: int = 8) -> Optional[Tuple[int, Tuple[float, ...]]]:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    try:
        im = Image.open(path).convert("RGB")
        g = np.asarray(im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.int16)
        diff = g[:, 1:] > g[:, :-1]
        bits = 0
        for b in diff.flatten().tolist():
            bits = (bits << 1) | (1 if b else 0)
        colour = tuple(np.asarray(im.resize((2, 2), Image.LANCZOS), dtype=np.float32).flatten().tolist())
        return bits, colour
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _colour_dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


def _group_by(n: int, is_dup) -> List[List[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if is_dup(i, j):
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def _collect_images(train_data_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(train_data_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def _remove_pair(image_path: str) -> None:
    base, _ = os.path.splitext(image_path)
    for p in [image_path] + [base + ext for ext in _SIDECAR_EXTS]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def dedup_dataset(train_data_dir: str, max_hamming: int = MAX_HAMMING,
                  colour_thresh: float = COLOUR_THRESH, min_n: int = MIN_N,
                  min_keep_frac: float = MIN_KEEP_FRAC, dry_run: bool = False) -> Dict:
    """Remove near-duplicates in place (or report only when dry_run=True).

    Returns {n_before, n_after, n_removed, n_groups, dup_rate, [removed], note}.
    Deterministic: the lexicographically-first member of each group is kept.
    """
    paths = _collect_images(train_data_dir)
    n = len(paths)
    result = {"n_before": n, "n_after": n, "n_removed": 0, "n_groups": n, "dup_rate": 0.0}
    if n < min_n:
        result["note"] = f"N={n} < min_n={min_n}; dedup skipped"
        return result

    sigs: List[Tuple[str, int, Tuple[float, ...]]] = []
    for p in paths:
        sig = _signature(p)
        if sig is not None:
            sigs.append((p, sig[0], sig[1]))
    if len(sigs) < min_n:
        result["note"] = "too few hashable images; dedup skipped"
        return result

    hpaths = [p for p, _, _ in sigs]
    hashes = [h for _, h, _ in sigs]
    colours = [c for _, _, c in sigs]
    groups = _group_by(len(sigs), lambda i, j:
                       _hamming(hashes[i], hashes[j]) <= max_hamming
                       and _colour_dist(colours[i], colours[j]) <= colour_thresh)
    result["n_groups"] = len(groups)

    redundant: List[str] = []
    for g in sorted(groups, key=len, reverse=True):
        for idx in g[1:]:
            redundant.append(hpaths[idx])

    floor = max(min_n, -(-int(len(sigs) * min_keep_frac)))  # ceil
    max_removable = max(0, len(sigs) - floor)
    to_remove = redundant[:max_removable]

    if not dry_run:
        for p in to_remove:
            _remove_pair(p)

    result["n_removed"] = len(to_remove)
    result["n_after"] = n - len(to_remove)
    result["dup_rate"] = round(len(to_remove) / n, 4) if n else 0.0
    result["removed"] = [os.path.basename(p) for p in to_remove]
    if len(redundant) > len(to_remove):
        result["note"] = f"keep-floor {floor} hit; {len(redundant) - len(to_remove)} dupes retained"
    return result
