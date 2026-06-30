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
from runtime import recipe as recipe_mod
from runtime import step_solver
from runtime import subtype_classifier


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


def write_config_skeleton(task_id: str, model_type: str, recipe: recipe_mod.Recipe) -> str:
    """Write a config skeleton to the path the toolchain expects.

    sd-scripts (sdxl/flux) consume TOML; ai-toolkit (qwen/z) consume YAML. In Fase 1
    the recipe is empty so this is a clearly-marked skeleton, NOT a runnable config.
    """
    is_toolkit = model_type in cst.AI_TOOLKIT_MODEL_TYPES
    ext = "yaml" if is_toolkit else "toml"
    config_path = paths.get_config_save_path(task_id, ext)
    paths.ensure_dirs(cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH)
    header = (
        f"# FASE 1 SKELETON — recipe NOT filled (Fase 2).\n"
        f"# model_type={model_type} toolchain={'ai-toolkit' if is_toolkit else 'sd-scripts'}\n"
        f"# missing recipe fields: {recipe_mod.missing_fields(recipe)}\n"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(header)
    log(f"Wrote config skeleton -> {config_path}")
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

    # 2. classify subtype -> bucket (structure; recipe mapping is Fase 2)
    bucket, confidence, sig = subtype_classifier.classify(images_dir, args.trigger_word)
    log(f"signature={sig.to_dict()}")
    log(f"bucket={bucket} confidence={confidence}")

    # 3. deterministic window + step plan (it/s measured live in Fase 2)
    plan = step_solver.build_plan(
        num_pairs=sig.image_count,
        model_type=args.model_type,
        hours_to_complete=args.hours_to_complete,
        it_per_s=None,
    )
    log(f"predicted_window_hours={plan.window_hours} (target_steps pending live it/s)")

    # 4. recipe (PLACEHOLDER in Fase 1)
    recipe = recipe_mod.resolve_recipe(args.model_type, bucket)
    config_path = write_config_skeleton(args.task_id, args.model_type, recipe)

    # 5. gate: do not launch a real run without a recipe
    if not recipe_mod.is_complete(recipe):
        log("RECIPE NOT FILLED — Fase 1 scaffold stops here (no training launched).")
        log(f"Would run: {' '.join(build_training_command(args.model_type, config_path))}")
        return 0

    # Fase 2 path (unreachable in Fase 1):
    cmd = build_training_command(args.model_type, config_path)
    log(f"Launching training: {' '.join(cmd)}")
    import subprocess
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
