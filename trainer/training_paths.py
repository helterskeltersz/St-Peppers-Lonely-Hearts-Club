"""Path helpers. Pure functions, no side effects beyond mkdir helpers."""

import os

from trainer import constants as cst


def get_image_base_model_path(model: str) -> str:
    """HF id -> local cache path. The validator pre-downloads to /cache/models
    with the HF "org/name" rewritten as "org--name"."""
    if os.path.isabs(model) and os.path.exists(model):
        return model
    safe = model.replace("/", "--")
    return os.path.join(cst.CACHE_MODELS_DIR, safe)


def get_dataset_zip_path(task_id: str) -> str:
    """Validator drops the dataset as /cache/datasets/{task_id}_tourn.zip."""
    return os.path.join(cst.CACHE_DATASETS_DIR, f"{task_id}{cst.DATASET_ZIP_SUFFIX}")


def get_image_training_images_dir(task_id: str) -> str:
    return os.path.join(cst.IMAGE_CONTAINER_IMAGES_PATH, task_id)


def get_config_save_path(task_id: str, ext: str) -> str:
    """ext = 'toml' (sd-scripts) or 'yaml' (ai-toolkit)."""
    return os.path.join(cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.{ext}")


def get_checkpoints_output_path(task_id: str, expected_repo_name: str) -> str:
    """Output the validator collects: /app/checkpoints/{task_id}/{expected_repo_name}."""
    return os.path.join(cst.OUTPUT_CHECKPOINTS_PATH, task_id, expected_repo_name or "output")


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
