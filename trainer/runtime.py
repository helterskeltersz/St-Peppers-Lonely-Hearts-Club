"""Dockerfile path resolution.

Mirrors the resolution contract of rayonlabs/G.O.D `trainer/runtime.py:get_dockerfile_path`
(verified in FASE0_VERIFY.md row h3): DEFAULT path under ops/docker/ is preferred,
the LEGACY dockerfiles/ path is accepted as a fallback, and a missing file raises
FileNotFoundError naming the preferred path. Keeping this contract identical means the
real validator resolver will also find our dockerfiles.
"""

import os

from trainer import constants as cst


def get_dockerfile_path(model_type: str, repo_root: str) -> str:
    """Return the absolute path to the dockerfile the validator would build for `model_type`.

    Raises FileNotFoundError if neither the preferred nor legacy path exists.
    """
    if model_type in cst.AI_TOOLKIT_MODEL_TYPES:
        candidates = cst.IMAGE_TOOLKIT_DOCKERFILE_PATHS
    else:
        candidates = cst.IMAGE_DOCKERFILE_PATHS

    for rel in candidates:
        full = os.path.join(repo_root, rel)
        if os.path.exists(full):
            return full

    raise FileNotFoundError(
        f"No dockerfile found for model_type={model_type!r}; expected one of: "
        f"{', '.join(candidates)}"
    )
