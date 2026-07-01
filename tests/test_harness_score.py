"""Unit tests for the scoring harness MATH (CPU, dummy arrays).

These assert the harness computes L2 + the 0.25/0.75 composite + aggregation +
held-out split correctly. They do NOT assert calibration: matching the validator's
absolute scale / exact reduction is a GPU step against CALIBRATION_ANCHOR.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import constants as C
from harness.score import pixel_l2, weighted_score, aggregate
from harness.evaluator import (EvalItem, holdout_split, seeds_for,
                               evaluate_checkpoint, compare_runs)


def _img(val, shape=(4, 4, 3)):
    return np.full(shape, val, dtype="float32")


# ---- pixel_l2 ----
def test_l2_identical_is_zero():
    a = _img(0.5)
    assert pixel_l2(a, a) == 0.0


def test_l2_known_values():
    a, b = _img(0.0), _img(1.0)
    assert pixel_l2(a, b, "mse") == pytest.approx(1.0)
    assert pixel_l2(a, b, "rmse") == pytest.approx(1.0)
    assert pixel_l2(a, b, "mae") == pytest.approx(1.0)


def test_l2_partial_diff():
    a, b = _img(0.0), _img(0.5)
    assert pixel_l2(a, b, "mse") == pytest.approx(0.25)
    assert pixel_l2(a, b, "rmse") == pytest.approx(0.5)
    assert pixel_l2(a, b, "mae") == pytest.approx(0.5)


def test_l2_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pixel_l2(_img(0.0, (4, 4, 3)), _img(0.0, (2, 2, 3)))


def test_l2_uint8_normalised():
    a = np.zeros((4, 4, 3), dtype="uint8")
    b = np.full((4, 4, 3), 255, dtype="uint8")
    assert pixel_l2(a, b, "mse") == pytest.approx(1.0)


# ---- weighting ----
def test_weights_sum_to_one():
    assert C.W_TEXT + C.W_NOTEXT == pytest.approx(1.0)


def test_weighted_score_formula():
    # 0.25*0.1 + 0.75*0.2 = 0.175
    assert weighted_score(0.1, 0.2) == pytest.approx(0.175)


def test_notext_dominates():
    # improving notext by X moves the score 3x more than the same move in text.
    base = weighted_score(0.2, 0.2)
    d_text = base - weighted_score(0.1, 0.2)
    d_notext = base - weighted_score(0.2, 0.1)
    assert d_notext == pytest.approx(3 * d_text)


def test_aggregate():
    r = aggregate([0.1, 0.3], [0.2, 0.2])
    assert r["text_l2"] == pytest.approx(0.2)
    assert r["notext_l2"] == pytest.approx(0.2)
    assert r["score"] == pytest.approx(0.2)


# ---- held-out split ----
def test_holdout_deterministic_and_disjoint():
    items = list(range(10))
    tr1, hd1 = holdout_split(items, frac=0.2, seed=42)
    tr2, hd2 = holdout_split(items, frac=0.2, seed=42)
    assert hd1 == hd2 and tr1 == tr2                 # deterministic
    assert set(tr1).isdisjoint(hd1)                  # no leakage
    assert sorted(tr1 + hd1) == items                # partition
    assert len(hd1) == 2


def test_holdout_min_one():
    _, hd = holdout_split(list(range(3)), frac=0.01)
    assert len(hd) >= 1


def test_seeds_for():
    assert seeds_for(42, 10) == list(range(42, 52))


# ---- denoise per arch ----
def test_denoise_per_arch():
    assert C.denoise_for("sdxl") == 0.9
    assert C.denoise_for("flux") == 0.75
    assert C.denoise_for("qwen-image") == 0.93
    assert C.denoise_for("z-image") == 0.90


# ---- evaluate_checkpoint with a dummy reconstruct_fn (no model) ----
def test_evaluate_perfect_reconstruction_scores_zero():
    items = [EvalItem(_img(0.5), "a prompt")]
    perfect = lambda orig, prompt, seed, denoise, mt: orig
    r = evaluate_checkpoint(items, "sdxl", perfect, n_seeds=3)
    assert r["score"] == pytest.approx(0.0)
    assert r["denoise"] == 0.9
    assert r["calibrated"] is False


def test_evaluate_notext_offset_weighted():
    # text pass reconstructs perfectly (L2=0); notext pass is off by 0.5 (mse=0.25).
    items = [EvalItem(_img(0.0), "p")]

    def recon(orig, prompt, seed, denoise, mt):
        return orig if prompt else _img(0.5)

    r = evaluate_checkpoint(items, "sdxl", recon, n_seeds=4)
    # score = 0.25*0 + 0.75*0.25 = 0.1875
    assert r["text_l2"] == pytest.approx(0.0)
    assert r["notext_l2"] == pytest.approx(0.25)
    assert r["score"] == pytest.approx(0.1875)


def test_compare_runs_lower_wins():
    a = {"score": 0.06490}
    b = {"score": 0.07098}
    c = compare_runs(a, b)
    assert c["winner"] == "a"
    assert c["delta"] == pytest.approx(0.00608)
