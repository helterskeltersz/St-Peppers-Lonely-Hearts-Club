"""Mirror of rayonlabs/G.O.D tests/trainer/test_runtime_dockerfile_paths.py.

Validates that our get_dockerfile_path honours the same resolution contract the real
validator uses: prefer ops/docker, fall back to legacy dockerfiles/, error when neither
exists. Plus an integration check that THIS repo's two dockerfiles actually resolve.

Run: python -m pytest tests/test_dockerfile_paths.py
"""

import os

import pytest

from trainer import constants as cst
from trainer.runtime import get_dockerfile_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_prefers_reorganized_ops_docker_path(tmp_path):
    preferred = tmp_path / cst.DEFAULT_IMAGE_DOCKERFILE_PATH
    legacy = tmp_path / cst.LEGACY_IMAGE_DOCKERFILE_PATH
    preferred.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    preferred.touch()
    legacy.touch()
    assert get_dockerfile_path(cst.MODEL_TYPE_SDXL, str(tmp_path)) == str(preferred)


def test_supports_legacy_image_path(tmp_path):
    legacy = tmp_path / cst.LEGACY_IMAGE_DOCKERFILE_PATH
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert get_dockerfile_path(cst.MODEL_TYPE_FLUX, str(tmp_path)) == str(legacy)


def test_supports_legacy_toolkit_path(tmp_path):
    legacy = tmp_path / cst.LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert get_dockerfile_path(cst.MODEL_TYPE_QWEN_IMAGE, str(tmp_path)) == str(legacy)


def test_errors_when_no_supported_path_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="standalone-image-trainer.dockerfile"):
        get_dockerfile_path(cst.MODEL_TYPE_SDXL, str(tmp_path))


@pytest.mark.parametrize(
    "model_type,expected_rel",
    [
        (cst.MODEL_TYPE_SDXL, cst.DEFAULT_IMAGE_DOCKERFILE_PATH),
        (cst.MODEL_TYPE_FLUX, cst.DEFAULT_IMAGE_DOCKERFILE_PATH),
        (cst.MODEL_TYPE_QWEN_IMAGE, cst.DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH),
        (cst.MODEL_TYPE_ZIMAGE, cst.DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH),
    ],
)
def test_this_repo_dockerfiles_resolve(model_type, expected_rel):
    resolved = get_dockerfile_path(model_type, REPO_ROOT)
    assert resolved == os.path.join(REPO_ROOT, expected_rel)
    assert os.path.exists(resolved)
