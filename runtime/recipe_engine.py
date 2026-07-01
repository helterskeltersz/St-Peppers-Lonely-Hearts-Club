"""Recipe engine v3 — 3-axis adaptive recipe with a table-heavy cascade.

    recipe = f(category, n_images, base_model, model_type)

Axes (VERIFIED against winner `zayden_recipe.py`):
  1. SHAPE   category -> subject (Prodigy) | style (AdamW)      [category_detection]
  2. SIZE    n_images -> talla bucket ECH/CH/M/G/EG, epoch|LR curve
  3. BASE    per-base-model plan table (cascade) + capacity preset

Sumbu 3 = CASCADE TABEL-HEAVY + 1 GUARD (locked in MODEL_PLANS_AUDIT.md, 306/307
rows clean). A listed base uses its real A/B-tuned table row; a sanity-guard
neutralises the single broken class (blue-pencil style/CH: LR 1.0 @ AdamW); an
unlisted/anonymised base or a guard-rejected row falls to the parametric generator.

SDXL/Flux use this module (kohya overlay dict). Qwen/Z-Image use
build_aitoolkit_config (F6/F7) — a separate config shape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from runtime import noise as noise_mod
from runtime import step_budget
from runtime.category_detection import category_shape

_DATA = Path(__file__).with_name("data") / "model_plans.json"

# ---- Sumbu 2: talla buckets (VERIFIED zayden_recipe.py:49-100) --------------
TALLAS = ("ECH", "CH", "M", "G", "EG")


def size_bucket(n: int) -> str:
    if n <= 10: return "ECH"
    if n <= 20: return "CH"
    if n <= 30: return "M"
    if n <= 50: return "G"
    return "EG"


_SUBJECT_BUCKETS = {
    "ECH": {"ep": 73, "b": 4, "gas": 1, "ulr": 0.95, "sched": "constant", "snr": 5, "no": 0.03, "dc": 1.2, "wd": 0.012},
    "CH":  {"ep": 33, "b": 4, "gas": 1, "ulr": 1.0, "sched": "constant", "snr": 5, "dc": 1.0, "wd": 0.008},
    "M":   {"ep": 26, "b": 8, "gas": 1, "ulr": 1.0, "sched": "constant_with_warmup", "warm": 50, "snr": 6, "dc": 1.1, "wd": 0.01},
    "G":   {"ep": 19, "b": 10, "gas": 1, "ulr": 1.1, "sched": "constant_with_warmup", "warm": 75, "snr": 6, "dc": 1.0, "wd": 0.01},
    "EG":  {"ep": 11, "b": 12, "gas": 1, "ulr": 1.2, "sched": "constant_with_warmup", "warm": 100, "snr": 6, "dc": 0.9, "wd": 0.01},
}
_STYLE_BUCKETS = {
    "ECH": {"ep": 28, "b": 4, "gas": 2, "ulr": 3.0e-5, "telr": 1.0e-6, "sched": "constant_with_warmup", "warm": 25, "snr": 6, "no": 0.02, "wd": 1.5e-4},
    "CH":  {"ep": 32, "b": 6, "gas": 1, "ulr": 2.8e-5, "telr": 1.2e-6, "sched": "cosine", "warm": 50, "snr": 6, "no": 0.025, "wd": 1.2e-4},
    "M":   {"ep": 24, "b": 8, "gas": 1, "ulr": 3.2e-5, "telr": 1.5e-6, "sched": "cosine", "warm": 75, "snr": 6, "no": 0.025, "wd": 1.0e-4},
    "G":   {"ep": 18, "b": 10, "gas": 1, "ulr": 3.6e-5, "telr": 1.8e-6, "sched": "cosine", "warm": 100, "snr": 6, "no": 0.035, "wd": 1.0e-4},
    "EG":  {"ep": 11, "b": 12, "gas": 1, "ulr": 4.0e-5, "telr": 2.0e-6, "sched": "cosine", "warm": 120, "snr": 6, "no": 0.04, "wd": 1.0e-4},
}

# ---- F9: capacity presets (VERIFIED zayden_recipe.py:108-137) ---------------
_CAP_DEFAULT = (32, 32, True)
_CAP_RANK64_CONV = {"SG161222/RealVisXL_V4.0", "GraydientPlatformAPI/albedobase2-xl",
    "dataautogpt3/ProteusV0.5", "femboysLover/RealisticStockPhoto-fp16", "zenless-lab/sdxl-blue-pencil-xl-v7"}
_CAP_RANK64_PLAIN = {"dataautogpt3/TempestV0.1", "mann-e/Mann-E_Dreams", "zenless-lab/sdxl-aam-xl-anime-mix"}
_CAP_RANK96_CONV = {"cagliostrolab/animagine-xl-4.0"}
_CAP_RANK32_PLAIN = {"John6666/hassaku-xl-illustrious-v10style-sdxl", "fluently/Fluently-XL-Final",
    "openart-custom/DynaVisionXL", "recoilme/colorfulxl", "zenless-lab/sdxl-anima-pencil-xl-v5",
    "zenless-lab/sdxl-anything-xl", "Corcelio/mobius", "OnomaAIResearch/Illustrious-xl-early-release-v0"}

# F8 generator: photoreal noisy bases prefer Huber (INFERRED from model_plans huber rows).
_PHOTOREAL_HINTS = ("realvis", "realistic", "realism", "dreamshaper", "leosams", "zavychroma",
                    "epic", "omnium", "protovision", "visionix", "albedo")
_ANIME_HINTS = ("anime", "animagine", "illustrious", "hassaku", "kohaku", "pencil", "anything",
                "nova-anime", "aam-xl", "art-diffusion")


def _capacity(model_name: str | None, is_style: bool):
    if is_style:
        return _CAP_DEFAULT
    name = (model_name or "").strip()
    if name in _CAP_RANK64_CONV: return (64, 64, True)
    if name in _CAP_RANK64_PLAIN: return (64, 64, False)
    if name in _CAP_RANK96_CONV: return (96, 96, True)
    if name in _CAP_RANK32_PLAIN: return (32, 32, False)
    return _CAP_DEFAULT


def _network_args(with_conv: bool) -> list:
    return ["conv_dim=4", "conv_alpha=4", "dropout=null"] if with_conv else []


def _family(model_name: str | None) -> str:
    n = (model_name or "").lower()
    if any(h in n for h in _ANIME_HINTS): return "anime"
    if any(h in n for h in _PHOTOREAL_HINTS): return "photoreal"
    return "other"


# ---- model_plans table loader ----------------------------------------------
_TABLE = None


def _table() -> dict:
    global _TABLE
    if _TABLE is None:
        try:
            _TABLE = json.loads(_DATA.read_text(encoding="utf-8"))
        except Exception:
            _TABLE = {"subject": {}, "style": {}}
    return _TABLE


# ---- sanity guard (Sumbu 3, neutralises blue-pencil-class bugs) -------------
def sanity_ok(optimizer: str, unet_lr) -> bool:
    """False => reject the row, fall back to generator. Catches an AdamW/Lion slot
    carrying a Prodigy-scale LR (blue-pencil style/CH: ulr=1.0 @ adamw)."""
    try:
        lr = float(unet_lr)
    except (TypeError, ValueError):
        return False
    if optimizer in {"adamw", "adamw8bit", "lion"} and lr > 1e-2:
        return False
    return True


def _row_to_overlay(row: dict, is_subject: bool) -> dict:
    o = {
        "max_train_epochs": row["ep"],
        "train_batch_size": row["b"],
        "gradient_accumulation_steps": row.get("gas", 1),
        "unet_lr": row["ulr"],
        "text_encoder_lr": row.get("telr", row["ulr"]),
        "lr_scheduler": row.get("sched", "constant"),
        "max_data_loader_n_workers": row.get("workers", 4),
    }
    if row.get("warm") is not None:
        o["lr_warmup_steps"] = row["warm"]
    if row.get("snr") is not None:
        o["min_snr_gamma"] = row["snr"]
    if row.get("no") is not None:
        o["noise_offset"] = row["no"]
        o["noise_offset_type"] = "Original"
    wd = row.get("wd")
    if is_subject:
        o["optimizer_type"] = "prodigy"
        o["optimizer_args"] = ["decouple=True", f"d_coef={row.get('dc',1.0)}",
            f"weight_decay={wd if wd is not None else 0.01}", "use_bias_correction=True", "safeguard_warmup=True"]
    else:
        o["optimizer_type"] = "adamw"
        o["optimizer_args"] = ["betas=(0.9, 0.999)", f"weight_decay={wd if wd is not None else 0.0001}", "eps=1e-08"]
    o.update(row.get("x", {}))
    return o


def _generator(is_subject: bool, talla: str, model_name: str | None) -> tuple[dict, str]:
    """Parametric fallback: global bucket + family multiplier + Huber for photoreal."""
    fam = _family(model_name)
    if is_subject:
        b = dict(_SUBJECT_BUCKETS[talla])
        o = {
            "max_train_epochs": b["ep"], "train_batch_size": b["b"], "gradient_accumulation_steps": b["gas"],
            "unet_lr": b["ulr"], "text_encoder_lr": b["ulr"], "lr_scheduler": b["sched"],
            "min_snr_gamma": b["snr"], "max_data_loader_n_workers": 4,
            "optimizer_type": "prodigy",
            "optimizer_args": ["decouple=True", f"d_coef={b['dc']}", f"weight_decay={b['wd']}",
                               "use_bias_correction=True", "safeguard_warmup=True"],
        }
        if b.get("warm"): o["lr_warmup_steps"] = b["warm"]
        if b.get("no") is not None:
            o["noise_offset"] = b["no"]; o["noise_offset_type"] = "Original"
        # F8: anime needs ~+48% epochs vs photoreal at equal size.
        if fam == "anime":
            o["max_train_epochs"] = round(b["ep"] * 1.48)
        # Huber for noisy photoreal bases (INFERRED from table huber rows).
        if fam == "photoreal":
            o.update({"loss_type": "huber", "huber_c": 0.1, "huber_schedule": "snr"})
    else:
        b = dict(_STYLE_BUCKETS[talla])
        o = {
            "max_train_epochs": b["ep"], "train_batch_size": b["b"], "gradient_accumulation_steps": b["gas"],
            "unet_lr": b["ulr"], "text_encoder_lr": b["telr"], "lr_scheduler": b["sched"],
            "lr_warmup_steps": b["warm"], "min_snr_gamma": b["snr"],
            "noise_offset": b["no"], "noise_offset_type": "Original",
            "optimizer_type": "adamw", "max_data_loader_n_workers": 4,
            "optimizer_args": ["betas=(0.9, 0.999)", f"weight_decay={b['wd']}", "eps=1e-08"],
        }
    return o, fam


def _caption_reg(is_subject: bool) -> dict:
    """F2/F3: keep trigger via keep_tokens=1 + shuffle; dropout 0.2 subject / 0.1 style."""
    return {"shuffle_caption": True, "keep_tokens": 1,
            "caption_dropout_rate": 0.2 if is_subject else 0.1}


def resolve_recipe(category: str, n_images: int, base_model: str | None,
                   model_type: str = "sdxl") -> dict:
    """Return a kohya config overlay (dict) for SDXL/Flux. `source` key records the
    cascade branch taken (table / table+guard-fallback / generator)."""
    shape = category_shape(category)
    is_subject = shape == "subject"
    talla = size_bucket(max(1, int(n_images or 0)))
    name = (base_model or "").strip()

    row = _table().get(shape, {}).get(name, {}).get(talla)
    if row:
        overlay = _row_to_overlay(row, is_subject)
        if sanity_ok(overlay.get("optimizer_type", ""), overlay.get("unet_lr")):
            source = "table"
        else:
            overlay, fam = _generator(is_subject, talla, name)
            source = f"generator(guard-reject;{fam})"
    else:
        overlay, fam = _generator(is_subject, talla, name)
        source = f"generator({fam})"

    # F9 capacity (unless a table x-override already pinned network_dim).
    dim, alpha, conv = _capacity(name, not is_subject)
    if "network_dim" not in overlay:
        overlay["network_dim"] = dim
        overlay["network_alpha"] = alpha
        overlay["network_args"] = _network_args(conv)

    # F5 noise per category (person/logo/social/design -> multires, clears offset).
    overlay.update(noise_mod.category_noise(category))
    # F2/F3 caption regularisation.
    for k, v in _caption_reg(is_subject).items():
        overlay.setdefault(k, v)

    overlay["_source"] = source
    overlay["_talla"] = talla
    overlay["_shape"] = shape
    return overlay


# ---- F6/F7: Qwen / Z-Image ai-toolkit config -------------------------------
_QWEN_RANK_MIN, _QWEN_RANK_MAX, _QWEN_RANK_REF = 120, 152, 134
_QWEN_LR_ANCHOR, _QWEN_WD_ANCHOR = 9.4e-5, 8.8e-6


def qwen_capacity(n_images: int) -> dict:
    n = max(1, int(n_images or 0))
    rank = int(min(_QWEN_RANK_MAX, max(_QWEN_RANK_MIN, round(118 + 0.85 * n))))
    lr = round(min(9.9e-5, max(8.6e-5, _QWEN_LR_ANCHOR * (_QWEN_RANK_REF / rank) ** 0.5)), 7)
    wd = round(min(1.05e-5, max(7.5e-6, _QWEN_WD_ANCHOR * (_QWEN_RANK_REF / rank))), 8)
    return {"rank": rank, "lr": lr, "weight_decay": wd}


def build_aitoolkit_config(model_type: str, n_images: int, category: str | None = None) -> dict:
    """Z-Image / Qwen-Image config (F6 capacity + F7 step budget). Runtime paths are
    filled by the caller."""
    mt = (model_type or "").lower()
    steps = step_budget.aitoolkit_steps(mt, n_images, category)
    if mt == "z-image":
        net = {"type": "lora", "linear": 32, "linear_alpha": 32, "conv": 16, "conv_alpha": 16}
        train = {"batch_size": 1, "steps": steps, "lr": 1.0e-4, "optimizer": "adamw8bit",
                 "noise_scheduler": "flowmatch", "timestep_type": "weighted", "dtype": "bf16"}
        model = {"arch": "zimage:turbo", "quantize": True, "qtype": "qfloat8",
                 "assistant_lora_path": "/cache/hf_cache/zimage_turbo_training_adapter_v2.safetensors"}
    elif mt == "qwen-image":
        cap = qwen_capacity(n_images)
        net = {"type": "lora", "linear": cap["rank"], "linear_alpha": cap["rank"]}
        train = {"batch_size": 1, "steps": steps, "lr": cap["lr"], "optimizer": "adamw8bit",
                 "noise_scheduler": "flowmatch", "timestep_type": "weighted", "dtype": "bf16",
                 "train_unet": True, "train_text_encoder": False,
                 "optimizer_params": {"weight_decay": float(cap["weight_decay"])},
                 "ema_config": {"use_ema": True, "ema_decay": 0.995}, "do_cfg": True, "cfg_scale": 6.0}
        model = {"arch": "qwen_image", "quantize": True, "qtype": "float8", "low_vram": True}
    else:
        return {}
    return {"model_type": mt, "steps": steps, "network": net, "train": train, "model": model,
            "_source": "aitoolkit"}
