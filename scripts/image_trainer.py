#!/usr/bin/env python3
"""Standalone image trainer entrypoint (Fase 2a-1).

Validator CLI (verified, trainer/runtime.py:174-190 @ rayonlabs/G.O.D):
    --task-id --model --dataset-zip --model-type --expected-repo-name
    --hours-to-complete [--trigger-word]

Flow: extract dataset -> classify subtype -> resolve recipe -> caption prep ->
generate real config -> step plan -> CUDA gate. On CPU (no GPU) it stops AFTER
config generation: training needs GPU (Fase 2a-2). It never fabricates throughput.

Watch-item (1): the step solver's source of truth is --hours-to-complete (which the
validator already includes the Qwen +0.5h in). The dataset-size window is logged only
as a SANITY cross-check; it is NOT used to compute steps -> no double-count.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import constants as cst
from trainer import training_paths as paths
from runtime import config_gen
from runtime import recipe as recipe_mod
from runtime import step_solver
from runtime import subtype_classifier


def log(msg: str) -> None:
    print(f"[image_trainer] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone image LoRA trainer (Gradients SN56)")
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True, choices=list(cst.MODEL_TYPES))
    p.add_argument("--expected-repo-name")
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    return p.parse_args()


def resolve_dataset_zip(task_id: str, dataset_zip_arg: str) -> str:
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


def prepare_captions(images_dir: str, recipe: recipe_mod.Recipe, trigger_word: str | None) -> int:
    """Apply §3a caption strategy on the .txt files (CPU-safe).

    drop_trigger_word -> strip the trigger token (concept hidden from the no-text pass).
    use_exact_caption  -> leave captions verbatim (logo: visible text must stay).
    """
    if recipe.use_exact_caption or not recipe.drop_trigger_word or not trigger_word:
        return 0
    pattern = re.compile(rf"\b{re.escape(str(trigger_word).strip())}\b", re.IGNORECASE)
    edited = 0
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            fp = os.path.join(root, f)
            try:
                with open(fp, encoding="utf-8") as fh:
                    txt = fh.read()
                new = re.sub(r"\s+", " ", pattern.sub("", txt)).strip(" ,")
                if new != txt:
                    with open(fp, "w", encoding="utf-8") as fh:
                        fh.write(new)
                    edited += 1
            except OSError:
                continue
    return edited


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> int:
    log("---STARTING IMAGE TRAINING (Fase 2a-1)---")
    args = parse_args()

    model_path = paths.get_image_base_model_path(args.model)
    output_dir = paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    paths.ensure_dirs(output_dir)
    log(f"model_type={args.model_type} model={args.model} -> {model_path}")

    # 1. dataset
    zip_path = resolve_dataset_zip(args.task_id, args.dataset_zip)
    images_dir = extract_dataset(zip_path, args.task_id)

    # 2. classify -> bucket (4 signals)
    bucket, confidence, sig = subtype_classifier.classify(images_dir, args.trigger_word)
    log(f"signature={sig.to_dict()}")
    log(f"bucket={bucket} confidence={confidence}")

    # 3. recipe
    recipe = recipe_mod.resolve_recipe(args.model_type, bucket)
    log(f"recipe complete={recipe_mod.is_complete(recipe)} "
        f"rank={recipe.rank} alpha={recipe.alpha} lr={recipe.learning_rate} "
        f"opt={recipe.optimizer} caption_dropout={recipe.caption_dropout_rate}")

    # 4. caption strategy (§3a)
    n_edited = prepare_captions(images_dir, recipe, args.trigger_word)
    log(f"captions edited (trigger drop): {n_edited}")

    # 5. real config
    ext = "yaml" if args.model_type in cst.AI_TOOLKIT_MODEL_TYPES else "toml"
    config_path = paths.get_config_save_path(args.task_id, ext)
    ctx = config_gen.ConfigContext(
        model_path=model_path,
        train_data_dir=images_dir,
        output_dir=output_dir,
        steps=recipe.max_train_steps,        # fallback ceiling; solver overrides on GPU
        trigger_word=args.trigger_word,
    )
    config_path, _ = config_gen.generate_config(recipe, args.model_type, ctx, config_path)
    log(f"Generated config -> {config_path} (ext .{ext})")

    # 6. step plan — SOURCE OF TRUTH = --hours-to-complete (validator already added Qwen +0.5h).
    sanity_window = step_solver.predict_window_hours(sig.image_count, args.model_type)
    log(f"hours_to_complete (authoritative)={args.hours_to_complete} ; "
        f"dataset-window sanity={sanity_window}h (NOT used for steps)")

    # 7. CUDA gate — training needs GPU (Fase 2a-2). On CPU we stop here, cleanly.
    if not cuda_available():
        log("CUDA absent -> config + plan prepared, training NOT launched (needs GPU, Fase 2a-2).")
        log(f"Would run: {' '.join(build_training_command(args.model_type, config_path))}")
        return 0

    # ---- Fase 2a-2 (GPU) path ----
    it_per_s = step_solver.measure_throughput()  # raises until wired to a live trainer source
    plan = step_solver.build_plan(sig.image_count, args.model_type, args.hours_to_complete, it_per_s)
    log(f"target_steps={plan.target_steps}")
    import subprocess
    return subprocess.run(build_training_command(args.model_type, config_path)).returncode


def build_training_command(model_type: str, config_path: str) -> list[str]:
    if model_type in cst.AI_TOOLKIT_MODEL_TYPES:
        return ["python3", os.path.join(cst.AI_TOOLKIT_DIR, "run.py"), config_path]
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


if __name__ == "__main__":
    raise SystemExit(main())
