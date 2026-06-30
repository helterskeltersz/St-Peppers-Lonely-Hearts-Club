#!/bin/bash
# Container ENTRYPOINT. Forwards the validator's CLI args to the trainer.
set -euo pipefail
echo "[run_image_trainer.sh] args: $*"
exec python3 /workspace/scripts/image_trainer.py "$@"
