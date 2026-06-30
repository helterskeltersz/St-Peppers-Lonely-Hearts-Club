# standalone-image-toolkit-trainer.dockerfile — Qwen-Image + Z-Image (Ostris ai-toolkit)
#
# Toolchain pins: MATRIKS_RECIPE_FINAL_v2.md §5b.
# Built FROM SOURCE: ai-toolkit cloned from main (NOT PyPI — PyPI is stale), diffusers
# built from source (Z-Image only merged recently, not in any wheel release yet).
#
# OFFLINE-PROOF (§5b, most critical gotcha): the container has NO internet at runtime.
# We bake the Z-Image turbo training adapter into the image and force HF offline so the
# job never tries to download at start.

# CUDA 12.8 runtime to match torch cu128 wheels.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_home

# --- pinned commits (status 📐: current HEAD, resolved 2026-06-30). Override via build args.
ARG AI_TOOLKIT_REPO=https://github.com/ostris/ai-toolkit.git
ARG AI_TOOLKIT_SHA=4e50535478d59a6e418c4e153c6daa908ad240c5
ARG DIFFUSERS_REPO=https://github.com/huggingface/diffusers.git
ARG DIFFUSERS_SHA=b549ca91ac4a31350511ee5625043e5dc922fe3a
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# Z-Image turbo training adapter v2 (~324MB) — baked so runtime stays offline (§5b).
ARG ZIMAGE_ADAPTER_REPO=ostris/zimage_turbo_training_adapter
ARG ZIMAGE_ADAPTER_FILE=zimage_turbo_training_adapter_v2.safetensors

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-dev build-essential \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# torch / torchvision pins (§5b: torch==2.9.1 torchvision==0.24.1, cu128).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.9.1 torchvision==0.24.1 --index-url ${TORCH_INDEX}

# diffusers from source, pinned (Z-Image support not yet in a wheel release).
RUN python3 -m pip install "git+${DIFFUSERS_REPO}@${DIFFUSERS_SHA}#egg=diffusers"

# ai-toolkit @ main, pinned.
RUN git clone ${AI_TOOLKIT_REPO} /app/ai-toolkit && \
    cd /app/ai-toolkit && git checkout ${AI_TOOLKIT_SHA} && \
    python3 -m pip install -r requirements.txt

# Quantization (§5b: optimum-quanto qfloat8 + ARA uint3) + runtime helpers + classifier deps.
# pytesseract -> ocr_area_ratio; open_clip_torch -> subject_variance CLIP path (proxy fallback
# if CLIP weights aren't baked — relevant since this image runs HF-offline at runtime).
RUN python3 -m pip install \
        optimum-quanto \
        "fastapi" "uvicorn" "pydantic" "Pillow==11.1.0" "numpy" "PyYAML" "huggingface_hub" \
        "pytesseract" "open_clip_torch"

# Bake the adapter into the image HF cache AND a stable local path, while online (build time).
RUN mkdir -p /app/adapters && python3 -c "\
from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='${ZIMAGE_ADAPTER_REPO}', filename='${ZIMAGE_ADAPTER_FILE}', local_dir='/app/adapters'); \
print('baked adapter ->', p)"

# Force offline AFTER everything is baked (runtime has no internet).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN mkdir -p /dataset/configs /dataset/images /dataset/outputs /app/checkpoints /workspace

COPY trainer  /workspace/trainer
COPY runtime  /workspace/runtime
COPY scripts  /workspace/scripts
RUN chmod +x /workspace/scripts/run_image_trainer.sh /workspace/scripts/image_trainer.py

ENV PYTHONPATH=/workspace
WORKDIR /workspace

# Fase 2 note: Qwen flow_shift is derived dynamically inside ai-toolkit
# (calculate_shift) — DO NOT set it manually. Base weights come from
# /cache/models/<org--model>; assistant_lora_path -> baked adapter above.

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]
