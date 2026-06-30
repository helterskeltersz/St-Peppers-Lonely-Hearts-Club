"""Path + dockerfile constants for the standalone image trainer.

Clean reimplementation (NOT copied from G.O.D / poseidon). Values mirror the
validator contract verified in FASE0_VERIFY.md so the container reads/writes the
exact paths the validator mounts.

Sources of truth (see FASE0_VERIFY.md):
- /cache (ro volume), /app/checkpoints/ (rw volume)  -> trainer/constants.py:41-43 @ rayonlabs/G.O.D
- dockerfile resolver DEFAULT=ops/docker, LEGACY=dockerfiles -> trainer/constants.py:2-8 @ rayonlabs/G.O.D
"""

# ---- model-type values (exact CLI choices, ref trainer/runtime.py:181 @ G.O.D) ----
MODEL_TYPE_SDXL = "sdxl"
MODEL_TYPE_FLUX = "flux"
MODEL_TYPE_QWEN_IMAGE = "qwen-image"
MODEL_TYPE_ZIMAGE = "z-image"

MODEL_TYPES = (MODEL_TYPE_SDXL, MODEL_TYPE_FLUX, MODEL_TYPE_QWEN_IMAGE, MODEL_TYPE_ZIMAGE)

# sd-scripts (kohya) handles these; ai-toolkit (Ostris) handles the rest.
SD_SCRIPTS_MODEL_TYPES = (MODEL_TYPE_SDXL, MODEL_TYPE_FLUX)
AI_TOOLKIT_MODEL_TYPES = (MODEL_TYPE_QWEN_IMAGE, MODEL_TYPE_ZIMAGE)

# ---- cache + output paths (validator-mounted volumes) ----
CACHE_ROOT_PATH = "/cache"
CACHE_MODELS_DIR = "/cache/models"
CACHE_DATASETS_DIR = "/cache/datasets"
HUGGINGFACE_CACHE_PATH = "/cache/hf_cache"
OUTPUT_CHECKPOINTS_PATH = "/app/checkpoints"
# Validator hands the dataset as a zip named "{task_id}_tourn.zip" inside /cache/datasets.
DATASET_ZIP_SUFFIX = "_tourn.zip"

# ---- in-container working paths ----
IMAGE_CONTAINER_CONFIG_SAVE_PATH = "/dataset/configs"
IMAGE_CONTAINER_IMAGES_PATH = "/dataset/images"

# ---- vendored / cloned toolchain locations (baked into the images) ----
SD_SCRIPTS_DIR = "/app/sd-scripts"      # kohya sd-scripts @ branch sd3
AI_TOOLKIT_DIR = "/app/ai-toolkit"      # Ostris ai-toolkit @ main

# ---- dockerfile resolver paths (DEFAULT preferred, LEGACY fallback) ----
DEFAULT_IMAGE_DOCKERFILE_PATH = "ops/docker/standalone-image-trainer.dockerfile"
DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH = "ops/docker/standalone-image-toolkit-trainer.dockerfile"
LEGACY_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH = "dockerfiles/standalone-image-toolkit-trainer.dockerfile"

IMAGE_DOCKERFILE_PATHS = (DEFAULT_IMAGE_DOCKERFILE_PATH, LEGACY_IMAGE_DOCKERFILE_PATH)
IMAGE_TOOLKIT_DOCKERFILE_PATHS = (DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH, LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH)

# Checkpoint cadence (verified eval saves every 250 steps; keep a window for upload).
CHECKPOINT_EVERY_N_STEPS = 250
