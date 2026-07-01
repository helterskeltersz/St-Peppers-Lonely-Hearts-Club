#!/usr/bin/env python3
"""Standalone image trainer entrypoint.

Invoked by the validator with the EXACT CLI verified in FASE0_VERIFY.md
(rayonlabs/G.O.D `trainer/runtime.py:174-190`):

    --task-id --model --dataset-zip --model-type --expected-repo-name
    --hours-to-complete [--trigger-word]

There is no --cache flag: /cache is a read-only mounted volume, /app/checkpoints is rw.

FASE 1 SCOPE: this wires the full flow (dataset -> classify -> step plan -> config
skeleton -> route to toolchain) but the RECIPE IS A PLACEHOLDER. When the recipe is
not yet filled (always, in Fase 1) we stop before launching a real training run and
exit cleanly, so the gate (build + CLI + paths) can be validated without producing a
garbage checkpoint. Recipe injection is Fase 2.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

# Make /workspace importable regardless of CWD (containers run from /workspace).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import constants as cst
from trainer import training_paths as paths
from runtime import step_solver
from runtime import dedup as dedup_mod
from runtime import category_detection as catdet
from runtime import caption as caption_mod
from runtime import recipe_engine

# Training is only launched when explicitly enabled (GPU phase). On the laptop/CPU
# the engine resolves + writes a config and stops, so routing can be validated for $0.
_ALLOW_TRAINING = os.getenv("ALLOW_TRAINING", "0").strip().lower() in {"1", "true", "yes", "on"}


def log(msg: str) -> None:
    print(f"[image_trainer] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone image LoRA trainer (Gradients SN56)")
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--dataset-zip", required=True, help="Path/name of the dataset zip in /cache/datasets")
    p.add_argument("--model-type", required=True, choices=list(cst.MODEL_TYPES))
    p.add_argument("--expected-repo-name", help="Output repo name under /app/checkpoints/{task_id}")
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    return p.parse_args()


def resolve_dataset_zip(task_id: str, dataset_zip_arg: str) -> str:
    """The validator passes a name; the file lives in /cache/datasets. Accept an
    absolute path, a bare name, or fall back to the verified {task_id}_tourn.zip."""
    candidates = []
    if os.path.isabs(dataset_zip_arg):
        candidates.append(dataset_zip_arg)
    candidates.append(os.path.join(cst.CACHE_DATASETS_DIR, os.path.basename(dataset_zip_arg)))
    candidates.append(paths.get_dataset_zip_path(task_id))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Dataset zip not found. Tried: {candidates}")


def extract_dataset(zip_path: str, task_id: str) -> str:
    images_dir = paths.get_image_training_images_dir(task_id)
    paths.ensure_dirs(images_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(images_dir)
    log(f"Extracted dataset -> {images_dir}")
    return images_dir


def _emit_toml(d: dict) -> str:
    """Minimal TOML emitter for a flat overlay dict (scalars/str/lists). Keys with a
    leading underscore are engine metadata and are written as comments."""
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return '"' + str(v).replace('"', '\\"') + '"'
    lines = []
    for k, v in d.items():
        if k.startswith("_"):
            lines.append(f"# {k} = {v}")
        else:
            lines.append(f"{k} = {fmt(v)}")
    return "\n".join(lines) + "\n"


def write_config(task_id: str, model_type: str, overlay: dict) -> str:
    """Write the resolved recipe. sd-scripts (sdxl/flux) => TOML overlay; ai-toolkit
    (qwen/z) => JSON config. Base-TOML merge + runtime paths are wired at the GPU phase."""
    is_toolkit = model_type in cst.AI_TOOLKIT_MODEL_TYPES
    ext = "json" if is_toolkit else "toml"
    config_path = paths.get_config_save_path(task_id, ext)
    paths.ensure_dirs(cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH)
    with open(config_path, "w", encoding="utf-8") as f:
        if is_toolkit:
            import json
            f.write(json.dumps(overlay, indent=2))
        else:
            f.write(f"# v3 recipe overlay ({model_type})\n")
            f.write(_emit_toml(overlay))
    log(f"Wrote config -> {config_path}")
    return config_path


def build_training_command(model_type: str, config_path: str) -> list[str]:
    """Route to the correct toolchain. STRUCTURE (not recipe): mirrors the proven
    launch shapes — accelerate+sd-scripts for sdxl/flux, ai-toolkit run.py for qwen/z."""
    if model_type in cst.AI_TOOLKIT_MODEL_TYPES:
        return ["python3", os.path.join(cst.AI_TOOLKIT_DIR, "run.py"), config_path]
    # sdxl -> sdxl_train_network.py, flux -> flux_train_network.py
    script = os.path.join(cst.SD_SCRIPTS_DIR, f"{model_type}_train_network.py")
    return [
        "accelerate", "launch",
        "--dynamo_backend", "no",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "2",
        script,
        "--config_file", config_path,
    ]


def main() -> int:
    log("---STARTING IMAGE TRAINING (Fase 1 scaffold)---")
    args = parse_args()

    model_path = paths.get_image_base_model_path(args.model)
    output_dir = paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    paths.ensure_dirs(output_dir)
    log(f"model_type={args.model_type} model={args.model} -> {model_path}")
    log(f"output_dir={output_dir} hours_to_complete={args.hours_to_complete}")

    # 1. dataset
    zip_path = resolve_dataset_zip(args.task_id, args.dataset_zip)
    images_dir = extract_dataset(zip_path, args.task_id)

    # 2. dedup (pre-caption) — drop near-duplicates so no frame gets Nx repeat weight
    dd = dedup_mod.dedup_dataset(images_dir)
    log(f"dedup: {dd.get('n_before')}->{dd.get('n_after')} removed={dd.get('n_removed')} "
        f"note={dd.get('note','')}")

    # 3. category (6-way keyword) + trigger recovery -> shape
    prompts = catdet.read_caption_prompts(images_dir)
    category = catdet.detect_category(prompts)
    trigger = args.trigger_word or catdet.detect_trigger(prompts)
    shape = catdet.category_shape(category)
    n_images = len([f for r, _, fs in os.walk(images_dir) for f in fs
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))])
    log(f"category={category} shape={shape} trigger={'yes' if trigger else 'no'} n={n_images}")

    # 4. captions — verbatim+tail (subject/text-heavy) | rewrite (style); trigger kept
    csum = caption_mod.build_captions(images_dir, category, args.model_type, trigger)
    log(f"caption: mode={csum.get('mode')} vlm={csum.get('vlm')} {csum.get('note','')}")

    # 5. deterministic window (upper cap) + step plan
    plan = step_solver.build_plan(num_pairs=n_images, model_type=args.model_type,
                                  hours_to_complete=args.hours_to_complete, it_per_s=None)
    log(f"predicted_window_hours={plan.window_hours} (target_steps pending live it/s)")

    # 6. resolve recipe (3-axis cascade) — sdxl/flux overlay OR qwen/z ai-toolkit config
    if args.model_type in cst.AI_TOOLKIT_MODEL_TYPES:
        overlay = recipe_engine.build_aitoolkit_config(args.model_type, n_images, category)
    else:
        overlay = recipe_engine.resolve_recipe(category, n_images, args.model, args.model_type)
    log(f"recipe source={overlay.get('_source')} talla={overlay.get('_talla')}")
    config_path = write_config(args.task_id, args.model_type, overlay)

    # 7. gate: launch training only when explicitly enabled (GPU phase)
    if not _ALLOW_TRAINING:
        log("ALLOW_TRAINING unset — config resolved & written; no training launched (CPU/validation mode).")
        log(f"Would run: {' '.join(build_training_command(args.model_type, config_path))}")
        return 0

    cmd = build_training_command(args.model_type, config_path)
    log(f"Launching training: {' '.join(cmd)}")
    import subprocess
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
