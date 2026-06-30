"""Config generator: Recipe -> sd-scripts TOML (sdxl/flux) or ai-toolkit YAML (qwen/z).

Watch-items (matriks Fase 2a-1) handled here:
  (1) hours_to_complete is the solver's source of truth — config-gen does NOT compute
      or add the Qwen +0.5h here (no double-count). `steps` is a fallback ceiling only.
  (2) Flux: cache_text_encoder_outputs = false (TE-cache conflicts with caption_dropout, §5b).
  (3) Qwen: discrete_flow_shift is NEVER emitted (ai-toolkit derives it internally).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from runtime.recipe import Recipe


@dataclass
class ConfigContext:
    model_path: str
    train_data_dir: str
    output_dir: str
    steps: int                      # fallback ceiling (solver overrides at runtime)
    trigger_word: str | None = None


# ---------- tiny TOML writer (flat key = value; sd-scripts configs are flat) ----------
def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace('"', '\\"') + '"'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {v!r}")


def to_toml(d: dict) -> str:
    return "".join(f"{k} = {_toml_scalar(v)}\n" for k, v in d.items())


# --------------------------- sd-scripts TOML (sdxl/flux) ---------------------------
def build_sd_scripts_config(r: Recipe, model_type: str, ctx: ConfigContext) -> dict:
    network_args: list[str] = []
    if r.conv is not None:  # LoCon: conv via network_args on networks.lora
        network_args += [f"conv_dim={r.conv}", f"conv_alpha={r.conv_alpha}"]

    cfg: dict = {
        "pretrained_model_name_or_path": ctx.model_path,
        "train_data_dir": ctx.train_data_dir,
        "output_dir": ctx.output_dir,
        "output_name": "last",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "mixed_precision": "bf16",
        "loss_type": "l2",
        "network_module": "networks.lora_flux" if model_type == "flux" else "networks.lora",
        "network_dim": r.rank,
        "network_alpha": r.alpha,
        "learning_rate": r.learning_rate,
        "unet_lr": r.unet_lr,
        "optimizer_type": r.optimizer,
        "lr_scheduler": r.scheduler,
        "resolution": r.resolution,
        "train_batch_size": r.train_batch_size,
        "max_train_steps": ctx.steps,            # (1) fallback ceiling; solver overrides at runtime
        "caption_extension": ".txt",
        "caption_dropout_rate": r.caption_dropout_rate,  # §3a lever #1
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "gradient_checkpointing": True,
    }
    if network_args:
        cfg["network_args"] = network_args
    if r.min_snr_gamma is not None:              # SDXL only (epsilon); flux omits it
        cfg["min_snr_gamma"] = r.min_snr_gamma
    if r.text_encoder_lr is not None:            # SDXL UNet-only -> None -> omit entirely
        cfg["text_encoder_lr"] = r.text_encoder_lr
    if r.train_unet_only is not None:            # SDXL True (drop both TEs); Flux False
        cfg["network_train_unet_only"] = r.train_unet_only
    if r.flip_aug is not None:                   # logo bucket -> False; else omit (trainer default)
        cfg["flip_aug"] = r.flip_aug

    if model_type == "flux":
        cfg["discrete_flow_shift"] = r.discrete_flow_shift  # ✅ Flux only
        cfg["timestep_sampling"] = r.timestep_sampling      # "sigmoid"
        cfg["model_prediction_type"] = "raw"
        cfg["guidance_scale"] = 1.0
        cfg["apply_t5_attn_mask"] = True
        # (2) TE-cache OFF so caption shuffle/dropout works (§5b).
        cfg["cache_text_encoder_outputs"] = False
    return cfg


# --------------------------- ai-toolkit YAML (qwen/z) ---------------------------
def build_ai_toolkit_config(r: Recipe, model_type: str, ctx: ConfigContext) -> dict:
    network: dict = {"type": "lora", "linear": r.rank, "linear_alpha": r.alpha}
    if r.conv is not None:                       # Z-Image has conv at baseline
        network["conv"] = r.conv
        network["conv_alpha"] = r.conv_alpha

    model_block: dict = {
        "name_or_path": ctx.model_path,
        "arch": r.arch,
        "quantize": bool(r.quantize),
        "qtype": r.qtype,
        "low_vram": True,
    }
    if r.assistant_lora_path:                    # Z-Image turbo adapter v2 (baked offline)
        model_block["assistant_lora_path"] = r.assistant_lora_path

    train_block: dict = {
        "batch_size": r.train_batch_size,
        "steps": ctx.steps,                      # (1) fallback ceiling; solver overrides at runtime
        "lr": r.learning_rate,
        "optimizer": r.optimizer,
        "noise_scheduler": r.noise_scheduler,    # flowmatch
        "timestep_type": r.timestep_type,        # weighted  (NOT sigmoid)
        "dtype": "bf16",
        "gradient_checkpointing": True,
        "train_unet": True,
        "train_text_encoder": False,
        "ema_config": {"use_ema": bool(r.ema)},  # §3a: EMA off
        # (3) NO flow_shift key — ai-toolkit derives it internally for qwen/z.
    }

    dataset_block: dict = {
        "folder_path": ctx.train_data_dir,
        "caption_ext": "txt",
        "cache_latents_to_disk": True,
        "resolution": [512, 768, 1024],
        "caption_dropout_rate": r.caption_dropout_rate,  # §3a lever #1 (ai-toolkit dataset block)
        "is_reg": False,
    }
    if r.flip_aug is not None:                   # logo -> flip_x: false (mirror corrupts glyphs)
        dataset_block["flip_x"] = r.flip_aug

    return {
        "job": "extension",
        "config": {
            "name": "last",
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": ctx.output_dir,
                    "device": "cuda",
                    "network": network,
                    "save": {
                        "dtype": "bf16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4,
                        "save_format": "diffusers",
                    },
                    "datasets": [dataset_block],
                    "train": train_block,
                    "model": model_block,
                }
            ],
        },
        "meta": {"name": f"{model_type}_lora", "version": "1.0"},
    }


def generate_config(r: Recipe, model_type: str, ctx: ConfigContext, save_path: str) -> tuple[str, str]:
    """Write the config file. Returns (path, text)."""
    is_toolkit = model_type in ("qwen-image", "z-image")
    if is_toolkit:
        import yaml  # PyYAML present in the toolkit image
        text = yaml.safe_dump(
            build_ai_toolkit_config(r, model_type, ctx), default_flow_style=False, sort_keys=False
        )
    else:
        text = to_toml(build_sd_scripts_config(r, model_type, ctx))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    return save_path, text
