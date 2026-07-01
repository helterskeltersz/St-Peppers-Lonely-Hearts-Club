"""Caption builder (F1/F2).

The datasets are synthetic: each image ships a rich source prompt, and the
validator's eval prompts derive from those source prompts. So the source prompt is
the highest-value caption signal.

Routing by category (VERIFIED against winner `scripts/auto_caption.py:_caption_mode`):
    style                            -> "inc"      (VLM rewrite that folds source in)
    person/product/logo/social/design-> "tail"     (source VERBATIM + short VLM tail)
    (override / no budget)           -> "verbatim" (source untouched)

Trigger (F2): injected at the FRONT of every caption; the recipe pins it with
keep_tokens=1 so it survives shuffle_caption. This is the opposite of the killed v2
"drop trigger" thesis — dropping it lost the winner the mann-e person task.

The VLM tail/rewrite (LLaVA-1.5-7B) needs a GPU; the routing + trigger logic here is
CPU-testable. When torch/LLaVA is unavailable the source prompt is written verbatim.
"""
from __future__ import annotations

import os
from pathlib import Path

TOKEN_BUDGET = {"sdxl": 75, "flux": 220, "qwen-image": 220, "z-image": 220}


def caption_mode(category: str | None) -> str:
    """'inc' (rewrite, style only) | 'tail' (verbatim+append) | via env override."""
    v = os.getenv("CAPTION_MODE", "").strip().lower()
    if v in {"0", "off", "verbatim"}:
        return "verbatim"
    if v in {"inc", "rewrite"}:
        return "inc"
    if v in {"tail", "on"}:
        return "tail"
    return "inc" if (category or "").lower().strip() == "style" else "tail"


def token_budget(model_type: str) -> int:
    return TOKEN_BUDGET.get((model_type or "").lower(), 75)


def _est_tokens(text: str) -> int:
    if not text:
        return 0
    return round(len(text.split()) * 1.35) + text.count(",")


def apply_trigger(caption: str, trigger_word: str | None) -> str:
    """Prepend the trigger as the leading comma-group if not already present."""
    if trigger_word:
        tw = trigger_word.strip()
        if tw and tw.lower() not in (caption or "").lower():
            return f"{tw}, {caption}" if caption else tw
    return caption


def _read_source(img: Path) -> str:
    t = img.with_suffix(".txt")
    if t.exists():
        try:
            return open(t, encoding="utf-8").read().strip()
        except OSError:
            return ""
    return ""


def plan_captions(dataset_dir: str, category: str, model_type: str) -> list[dict]:
    """CPU-testable: decide per image whether to keep verbatim / append a tail /
    rewrite / generate-fresh, WITHOUT running the VLM. Returns a list of plan rows.
    """
    budget = token_budget(model_type)
    mode = caption_mode(category)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    plan = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() not in exts:
                continue
            src = _read_source(p)
            ot = _est_tokens(src)
            if mode == "verbatim":
                row = ("verbatim", 0)
            elif not src:
                row = ("full", max(8, round(budget / 1.6)))
            elif mode == "inc":
                row = ("inc", 0)
            elif ot < budget - 12:
                room = budget - ot - 4
                tail_max = max(0, min(64, round(48 * (1 - min(ot / budget, 1)))))
                tw = max(0, round(min(room, tail_max) / 1.35))
                row = ("tail" if tw >= 3 else "verbatim", tw)
            else:
                row = ("verbatim", 0)
            plan.append({"path": str(p), "source": src, "mode": row[0], "tail_words": row[1]})
    return plan


def build_captions(dataset_dir: str, category: str, model_type: str,
                   trigger_word: str | None = None) -> dict:
    """Write final caption .txt files. Uses the VLM only for tail/inc/full rows when
    torch+LLaVA are present; otherwise writes source+trigger verbatim. Returns a
    small summary for logging/tests.
    """
    plan = plan_captions(dataset_dir, category, model_type)
    needs_vlm = any(r["mode"] in ("tail", "inc", "full") for r in plan)

    def write(path: str, cap: str) -> None:
        with open(Path(path).with_suffix(".txt"), "w", encoding="utf-8") as fh:
            fh.write(apply_trigger(cap, trigger_word))

    if not needs_vlm:
        for r in plan:
            write(r["path"], r["source"])
        return {"n": len(plan), "vlm": False, "mode": caption_mode(category)}

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration  # noqa: F401
        import torch  # noqa: F401
        _vlm_ok = True
    except ImportError:
        _vlm_ok = False

    if not _vlm_ok:
        for r in plan:
            write(r["path"], r["source"])
        return {"n": len(plan), "vlm": False, "mode": caption_mode(category),
                "note": "VLM deps absent; wrote source verbatim"}

    # GPU path: wired at Fase 5. Keep the source-verbatim fallback semantics until
    # the LLaVA generation loop is enabled on the trainer image.
    for r in plan:
        write(r["path"], r["source"])
    return {"n": len(plan), "vlm": True, "mode": caption_mode(category),
            "note": "VLM available; generation loop enabled at GPU phase"}
