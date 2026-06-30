# St. Pepper's Lonely Hearts Club — Gradients.io image trainer (SN56)

Standalone image-LoRA training repo for the Gradients.io (G.O.D / Bittensor SN56)
image tournament. Built clean from scratch — every line auditable — against the
verified validator contract, NOT a fork of a sanitised champion repo.

**Status: Fase 1 — validator gate + runtime scaffold. No recipe yet (Fase 2).**

## What the validator does

Clones this repo, resolves a dockerfile by `--model-type`, builds it, and runs the
container with (verified CLI, `trainer/runtime.py:174-190` @ rayonlabs/G.O.D):

```
--task-id --model --dataset-zip --model-type --expected-repo-name --hours-to-complete [--trigger-word]
```

`/cache` is mounted read-only (datasets at `/cache/datasets/{task_id}_tourn.zip`,
models at `/cache/models/<org--model>`); output goes to
`/app/checkpoints/{task_id}/{expected_repo_name}`.

## Layout

| Path | Role |
|---|---|
| `ops/docker/standalone-image-trainer.dockerfile` | SDXL + Flux (kohya `sd-scripts` @ `sd3`, from source) |
| `ops/docker/standalone-image-toolkit-trainer.dockerfile` | Qwen-Image + Z-Image (Ostris `ai-toolkit` @ `main`, offline-proof) |
| `scripts/run_image_trainer.sh` | container ENTRYPOINT |
| `scripts/image_trainer.py` | parses the CLI, routes by model-type |
| `trainer/` | paths, constants, dockerfile resolver |
| `runtime/subtype_classifier.py` | dataset signature → bucket (structure only) |
| `runtime/step_solver.py` | deterministic window → target steps |
| `runtime/recipe.py` | **placeholder** — Fase 2 fills the numbers |
| `miner/asgi.py` | port 7999 endpoint (gate) |
| `LICENSE.md`, `NOTICE` | byte-match the base (`6ad6353e…` / `3e316950…`) |

## Toolchain pins (MATRIKS §5b)

| | kohya `sd-scripts` | ai-toolkit | diffusers |
|---|---|---|---|
| ref | branch `sd3` @ `b8d1eb06…` | `main` @ `4e505354…` | `main` @ `b549ca91…` |
| torch | 2.6.0 / tv 0.21.0 (cu124) | 2.9.1 / tv 0.24.1 (cu128) | from source |

SHAs are current-HEAD pins (resolved 2026-06-30); override with build args, e.g.
`--build-arg SD_SCRIPTS_SHA=<sha>`.

## Test

```
pip install -r requirements-dev.txt
python -m pytest          # dockerfile path resolver contract
```

See `FASE1_REPORT.md` for the gate results and the open items handed to Fase 2.
