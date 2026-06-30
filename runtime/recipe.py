"""Recipe resolver (Fase 2a-1).

Composition order (matriks MATRIKS_RECIPE_FINAL_v2.md):
  1. §1 per-arch baseline       (✅ VERIFIED — read from core/training_templates/*)
  2. §3a unconditional override (📐 INFERRED — our decision; A/B in Fase 2a-2)
  3. §3b/§3c bucket delta        (📐 INFERRED — aggressive / prior / logo)

Status tags on every value:
  ✅ = verified baseline from G.O.D source.
  📐 = our strategy decision (sweepable; not yet A/B-confirmed).

NOTE: step count is NOT hardcoded here — it comes from the window-aware solver at
runtime (needs live GPU it/s). `max_train_steps` below is only a CPU/no-GPU fallback
ceiling so a config is always complete; the solver overrides it in Fase 2a-2.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from runtime.subtype_classifier import BUCKET_AGGRESSIVE, BUCKET_PRIOR, BUCKET_LOGO

# §3a lever #1. Default 0.30, sweepable up to 0.50 (matriks §3a/§3b).
DEFAULT_CAPTION_DROPOUT = 0.30
CAPTION_DROPOUT_RANGE = (0.30, 0.50)

# Logo bucket caption_dropout is PER-ARCH (📐), not flat. Rationale (also in _apply_bucket):
# the metric is pixel-L2 (not OCR), dominated by low-freq that transfers to the unconditional
# path -> moderate dropout, tuned to denoise strength (higher denoise = conditioning matters
# more = lower dropout). Higher-denoise arches (Qwen 0.93 / Z 0.90) keep more caption signal.
LOGO_CAPTION_DROPOUT = {
    "qwen-image": 0.15,  # 📐 denoise 0.93 (highest) -> lowest dropout
    "z-image": 0.20,     # 📐 denoise 0.90
    "flux": 0.25,        # 📐 denoise 0.75
    "sdxl": 0.30,        # 📐 denoise 0.90 but epsilon model -> keep at §3a default
}


@dataclass
class Recipe:
    # --- network ---
    rank: int | None = None
    alpha: int | None = None
    conv: int | None = None            # conv_dim (LoCon); None = conv off
    conv_alpha: int | None = None
    network_kind: str | None = None    # "lora" | "locon"
    # --- optimizer / lr ---
    learning_rate: float | None = None
    unet_lr: float | None = None
    text_encoder_lr: float | None = None
    optimizer: str | None = None
    scheduler: str | None = None
    # --- diffusion specifics ---
    min_snr_gamma: float | None = None       # SDXL only (epsilon); ignored by flow-matching
    discrete_flow_shift: float | None = None # Flux only; NEVER set for Qwen (toolkit derives it)
    timestep_sampling: str | None = None     # Flux sd-scripts: "sigmoid"
    timestep_type: str | None = None         # ai-toolkit (qwen/z): "weighted"
    noise_scheduler: str | None = None       # "flowmatch" for flow-matching arches
    # --- data / training ---
    resolution: str | None = None
    train_batch_size: int | None = None
    repeats: int | None = None
    max_train_steps: int | None = None       # FALLBACK; solver overrides at runtime
    train_unet_only: bool | None = None      # SDXL: True (drop both TEs); Flux: False
    flip_aug: bool | None = None             # None = trainer default; logo bucket -> False
    # --- §3a unconditional strategy ---
    caption_dropout_rate: float | None = None
    drop_trigger_word: bool | None = None
    ema: bool | None = None                  # always off (matriks §3a)
    use_exact_caption: bool | None = None    # logo bucket: keep visible text verbatim
    high_noise_bias: bool | None = None      # eval denoises from high noise -> bias training there
    # --- toolchain hints ---
    cache_text_encoder_outputs: bool | None = None  # Flux: MUST be False (TE-cache vs dropout, §5b)
    arch: str | None = None
    quantize: bool | None = None
    qtype: str | None = None
    assistant_lora_path: str | None = None


# ---------------------------------------------------------------------------
# §1 baselines — ✅ VERIFIED from core/training_templates/* (FASE0_VERIFY.md).
# ---------------------------------------------------------------------------
def _baseline(model_type: str) -> Recipe:
    if model_type == "sdxl":
        return Recipe(
            rank=32,                  # ✅ network_dim base_diffusion_sdxl.toml:37
            alpha=16,                 # ✅ network_alpha :35
            network_kind="lora",      # ✅ networks.lora :38
            learning_rate=1e-5,       # ✅ :20  (NOT 1e-4 — old claim KILLED)
            unet_lr=1e-5,             # ✅ :58
            text_encoder_lr=None,     # ✅ DECISION: SDXL UNet-only ALL buckets -> no TE LR.
                                      #    Two SDXL TEs destabilize training + the 0.75 no-text
                                      #    strategy discards the conditional path -> TE training is
                                      #    wasted. (consistent matriks §3)
            optimizer="AdamW8Bit",    # ✅ :42
            scheduler="constant",     # ✅ :22
            min_snr_gamma=5,          # ✅ :33 (active for SDXL)
            resolution="1024,1024",   # ✅ :47
            train_batch_size=4,       # ✅ :55
            repeats=10,               # ✅ DIFFUSION_SDXL_REPEATS trainer/constants.py:63
            max_train_steps=1600,     # ✅ :31 (fallback ceiling)
            train_unet_only=True,     # ✅ DECISION: SDXL UNet-only (all buckets)
        )
    if model_type == "flux":
        return Recipe(
            rank=128,                 # ✅ network_dim base_diffusion_flux.toml:40
            alpha=128,                # ✅ network_alpha :38
            network_kind="lora",      # ✅ networks.lora_flux :41
            learning_rate=5e-5,       # ✅ unet_lr :62
            unet_lr=5e-5,             # ✅ :62
            text_encoder_lr=5e-5,     # ✅ :58 ([5e-5,5e-5])
            optimizer="Adafactor",    # ✅ :44
            scheduler="constant",     # ✅ :26
            discrete_flow_shift=3.1582,  # ✅ :9
            timestep_sampling="sigmoid", # ✅ :59
            noise_scheduler="flowmatch", # ✅ (flux flow-matching)
            resolution="1024,1024",   # ✅ :49
            train_batch_size=1,       # ✅ :60
            repeats=1,                # ✅ DIFFUSION_FLUX_REPEATS trainer/constants.py:64
            max_train_steps=3000,     # ✅ :33 (fallback ceiling)
            train_unet_only=False,    # ✅ Flux: TE trains (TE-cache OFF for caption_dropout, §5b)
        )
    if model_type == "qwen-image":
        return Recipe(
            rank=32,                  # ✅ linear base_diffusion_qwen_image.yaml:10
            alpha=32,                 # ✅ linear_alpha :11
            network_kind="lora",      # ✅ :9
            learning_rate=1e-4,       # ✅ :26
            optimizer="adamw8bit",    # ✅ :27
            timestep_type="weighted", # ✅ :30 (NOT sigmoid)
            noise_scheduler="flowmatch",  # ✅ :29
            train_batch_size=1,       # ✅ :24
            max_train_steps=3000,     # ✅ :25 (fallback ceiling)
            arch="qwen_image",        # ✅ :37
            quantize=True,            # ✅ :38
            qtype="uint3",            # ✅ :39 (ARA low-bit)
            # discrete_flow_shift INTENTIONALLY unset — ai-toolkit derives it (matriks §6 #2).
        )
    if model_type == "z-image":
        return Recipe(
            rank=32,                  # ✅ linear base_diffusion_zimage.yaml:10
            alpha=32,                 # ✅ linear_alpha :11
            conv=16,                  # ✅ conv :12 (only arch with conv at baseline)
            conv_alpha=16,            # ✅ conv_alpha :13
            network_kind="locon",     # ✅ has conv
            learning_rate=1e-4,       # ✅ :28
            optimizer="adamw8bit",    # ✅ :29
            timestep_type="weighted", # ✅ :32
            noise_scheduler="flowmatch",  # ✅ :31
            train_batch_size=1,       # ✅ :26
            max_train_steps=2000,     # ✅ :27 (fallback ceiling)
            arch="zimage:turbo",      # ✅ :36
            quantize=True,            # ✅ :38
            qtype="qfloat8",          # ✅ :38-39
            assistant_lora_path=(    # ✅ :41 (adapter v2 baked into the toolkit image)
                "ostris/zimage_turbo_training_adapter/"
                "zimage_turbo_training_adapter_v2.safetensors"
            ),
        )
    raise ValueError(f"unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# §3a unconditional override — 📐 applied to ALL arches.
# 75% of score comes from the no-text pass, so push concepts onto the null path.
# ---------------------------------------------------------------------------
def _apply_unconditional(r: Recipe, caption_dropout: float) -> Recipe:
    r.caption_dropout_rate = caption_dropout  # 📐 §3a lever #1 (default 0.30)
    r.drop_trigger_word = True                # 📐 §3a: trigger gates knowledge absent in no-text pass
    r.ema = False                             # 📐 §3a: allow controlled overfit
    r.high_noise_bias = True                  # 📐 §3a: eval denoises from high noise (0.75-0.93)
    return r


# ---------------------------------------------------------------------------
# §3b / §3c bucket deltas — 📐 on top of baseline + override.
# Values are starting points (sweepable), not final.
# ---------------------------------------------------------------------------
def _apply_bucket(r: Recipe, model_type: str, bucket: str) -> Recipe:
    if bucket == BUCKET_AGGRESSIVE:
        # Overfit fast: fewer steps, checkpoint earlier (solver handles cadence).
        if model_type == "sdxl":
            r.min_snr_gamma = 1     # 📐 §3b: drop from 5 -> ~1 for faster fit
            r.conv = None           # 📐 conv off
            r.conv_alpha = None
            r.network_kind = "lora"
            r.learning_rate = r.unet_lr = 2e-5  # 📐 §3c: bump 1e-5 -> 2e-5 (low end; sweep to 5e-5)
        # flux/qwen/z: baseline rank/LR already suit aggressive; dropout 0.30 (override).
    elif bucket == BUCKET_PRIOR:
        # Prior-preserving: use whole window, slightly higher dropout, texture conv on.
        r.caption_dropout_rate = max(r.caption_dropout_rate or 0.0, 0.40)  # 📐 §3b: 0.40-0.50
        if model_type == "sdxl":
            r.min_snr_gamma = 5     # 📐 keep baseline (prior wants stability)
            r.conv = 16             # 📐 §3b: conv ON (LoCon) for texture
            r.conv_alpha = 16
            r.network_kind = "locon"
        if model_type in ("qwen-image", "z-image"):
            pass  # steps↑ handled by solver using full window
    elif bucket == BUCKET_LOGO:
        # Logo/text: caption carries the visible text. dropout is PER-ARCH (📐) — pixel-L2
        # metric is low-freq-dominated, so moderate dropout tuned to denoise strength
        # (higher denoise -> conditioning matters more -> lower dropout). Exact caption kept,
        # trigger KEPT (wins the 25% text term — opposite of other buckets), conv for edges.
        r.use_exact_caption = True       # 📐 §3c
        r.drop_trigger_word = False      # 📐 logo KEEPS trigger (other buckets drop it)
        r.caption_dropout_rate = LOGO_CAPTION_DROPOUT[model_type]  # 📐 per-arch (A/B wajib)
        r.resolution = "1024,1024"       # 📐 §3c: don't downscale text
        r.flip_aug = False               # ✅ logo: mirror corrupts asymmetric glyphs (all arch)
        if model_type == "sdxl":
            r.conv = 16                  # 📐 §3c: conv ON (edges)
            r.conv_alpha = 16
            r.network_kind = "locon"
        # qwen: rely on native text rendering — don't over-engineer the network (matriks §3c).
    else:
        raise ValueError(f"unknown bucket: {bucket!r}")
    return r


def resolve_recipe(model_type: str, bucket: str, caption_dropout: float | None = None) -> Recipe:
    """Compose baseline -> unconditional override -> bucket delta.

    caption_dropout lets Fase 2b sweep the §3a lever ({0.10,0.20,0.30,0.40}) per-call
    WITHOUT touching code. When passed, it overrides the bucket default (incl. logo per-arch).
    """
    r = _baseline(model_type)
    r = _apply_unconditional(r, DEFAULT_CAPTION_DROPOUT)
    r = _apply_bucket(r, model_type, bucket)        # may set bucket/per-arch dropout
    if caption_dropout is not None:
        r.caption_dropout_rate = caption_dropout    # explicit sweep override wins last
    return r


def is_complete(recipe: Recipe) -> bool:
    """Training-critical fields present. True once a recipe is resolved (Fase 2a-1)."""
    required = ("rank", "alpha", "learning_rate", "optimizer", "caption_dropout_rate")
    return all(getattr(recipe, name) is not None for name in required)


def missing_fields(recipe: Recipe) -> list[str]:
    return [f.name for f in fields(recipe) if getattr(recipe, f.name) is None]
