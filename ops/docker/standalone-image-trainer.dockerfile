# standalone-image-trainer.dockerfile — SDXL + Flux (kohya sd-scripts)
#
# Toolchain pins: MATRIKS_RECIPE_FINAL_v2.md §5b.
# Built FROM SOURCE (not a prebuilt base image) so every layer is auditable and
# reproducible when the validator re-clones + rebuilds.
#
# Layout (entrypoint shape, mkdir set) cross-validated against the proven poseidon
# repo; build CONTENT is our own per §5b.

# CUDA 12.4 runtime to match torch cu124 wheels.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- pinned commit (status 📐: current HEAD of branch sd3, resolved 2026-06-30).
#     Override at build: --build-arg SD_SCRIPTS_SHA=<sha> to lock an exact date.
#     Branch sd3 (NOT main): Flux/SD3 live on sd3; main has not merged Flux. (§5b)
ARG SD_SCRIPTS_REPO=https://github.com/kohya-ss/sd-scripts.git
ARG SD_SCRIPTS_SHA=b8d1eb067eba32bb105984678b97f05b11452940
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# torch / torchvision pins (§5b: torch==2.6.0 torchvision==0.21.0, cu124).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.6.0 torchvision==0.21.0 --index-url ${TORCH_INDEX}

# kohya sd-scripts @ sd3, pinned.
RUN git clone ${SD_SCRIPTS_REPO} /app/sd-scripts && \
    cd /app/sd-scripts && git checkout ${SD_SCRIPTS_SHA} && \
    python3 -m pip install -r requirements.txt

# LyCORIS for DoRA / LoKr / LoHa / LoCon (§5b: lycoris-lora==3.3.0).
# NOTE (§5b): LyCORIS 3.x changed wd_on_output default — Fase 2 must verify DoRA/LoKr
# checkpoints actually load at inference before relying on them.
RUN python3 -m pip install "lycoris-lora==3.3.0"

# Runtime helper deps used by the entrypoint / server.
RUN python3 -m pip install \
        "fastapi" "uvicorn" "pydantic" "Pillow==11.1.0" "numpy" "PyYAML" "toml" "huggingface_hub"

# Working dirs (validator mounts /cache ro and /app/checkpoints rw at run time).
RUN mkdir -p /dataset/configs /dataset/images /dataset/outputs /app/checkpoints /workspace

# Source overlay.
COPY trainer  /workspace/trainer
COPY runtime  /workspace/runtime
COPY scripts  /workspace/scripts
RUN chmod +x /workspace/scripts/run_image_trainer.sh /workspace/scripts/image_trainer.py

ENV PYTHONPATH=/workspace
WORKDIR /workspace

# Flux gotcha (§5b): --cache_text_encoder_outputs disables caption shuffle/dropout and
# forces network_train_unet_only. Because our strategy needs aggressive caption_dropout
# (the 0.75 no-text term), Fase 2 should keep TE-cache OFF for Flux. Config-level, not here.

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]
