"""Tests des métriques de localisation (Grad-CAM validé quantitativement).

Ces métriques portent une conclusion du rapport — « le modèle vise la lésion
dans X % des cas ». Une erreur de calcul y passerait inaperçue, d'où ces
vérifications sur des cas dont on connaît la réponse exacte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain.localization import (
    coverage,
    evaluate_localization,
    iou_score,
    localization_report,
    pointing_game,
    threshold_cam,
)


@pytest.fixture
def square_mask():
    """Masque carré 20×20 au centre d'une image 100×100 (4 % de la surface)."""
    m = np.zeros((100, 100), dtype=np.uint8)
    m[40:60, 40:60] = 1
    return m


@pytest.fixture
def centered_cam():
    """Carte gaussienne centrée sur la lésion."""
    y, x = np.mgrid[0:100, 0:100]
    return np.exp(-(((x - 50) ** 2 + (y - 50) ** 2) / (2 * 12.0 ** 2)))


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
def test_iou_identical_masks_is_one(square_mask):
    assert iou_score(square_mask, square_mask) == 1.0


def test_iou_disjoint_masks_is_zero(square_mask):
    other = np.zeros((100, 100), dtype=np.uint8)
    other[0:10, 0:10] = 1
    assert iou_score(other, square_mask) == 0.0


def test_iou_known_value():
    """Deux carrés 10×10 décalés de 5 : intersection 25, union 175."""
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    b = np.zeros((50, 50), dtype=np.uint8); b[15:25, 15:25] = 1
    assert abs(iou_score(a, b) - 25 / 175) < 1e-9


def test_iou_empty_masks_is_nan():
    z = np.zeros((10, 10), dtype=np.uint8)
    assert np.isnan(iou_score(z, z))


def test_iou_shape_mismatch_raises():
    with pytest.raises(ValueError, match="incompatibles"):
        iou_score(np.zeros((10, 10)), np.zeros((20, 20)))


# ---------------------------------------------------------------------------
# Pointing game
# ---------------------------------------------------------------------------
def test_pointing_hit_when_peak_inside(square_mask, centered_cam):
    assert pointing_game(centered_cam, square_mask) == 1.0


def test_pointing_miss_when_peak_outside(square_mask):
    cam = np.zeros((100, 100)); cam[5, 5] = 1.0
    assert pointing_game(cam, square_mask) == 0.0


def test_pointing_empty_mask_is_nan(centered_cam):
    assert np.isnan(pointing_game(centered_cam, np.zeros((100, 100))))


def test_pointing_insensitive_to_cam_scale(square_mask, centered_cam):
    """Le pointing game ne dépend que de la position du maximum."""
    assert pointing_game(centered_cam * 1000, square_mask) == \
           pointing_game(centered_cam, square_mask)


# ---------------------------------------------------------------------------
# Seuillage
# ---------------------------------------------------------------------------
def test_percentile_threshold_keeps_expected_fraction(centered_cam):
    binary = threshold_cam(centered_cam, "percentile", 80.0)
    frac = binary.mean()
    assert 0.18 < frac < 0.22, f"attendu ~20 % de pixels retenus, obtenu {frac:.1%}"


def test_percentile_is_scale_invariant(centered_cam):
    """Propriété essentielle : les résultats restent comparables d'une image
    à l'autre malgré des amplitudes d'activation différentes."""
    a = threshold_cam(centered_cam, "percentile", 80.0)
    b = threshold_cam(centered_cam * 57.3, "percentile", 80.0)
    assert np.array_equal(a, b)


def test_absolute_threshold_is_not_scale_invariant(centered_cam):
    a = threshold_cam(centered_cam, "absolute", 0.5)
    b = threshold_cam(centered_cam * 0.1, "absolute", 0.5)
    assert not np.array_equal(a, b)


def test_invalid_percentile_raises(centered_cam):
    with pytest.raises(ValueError):
        threshold_cam(centered_cam, "percentile", 150.0)


# ---------------------------------------------------------------------------
# Couverture
# ---------------------------------------------------------------------------
def test_coverage_full_when_prediction_contains_lesion(square_mask):
    everything = np.ones((100, 100), dtype=np.uint8)
    assert coverage(everything, square_mask) == 1.0


def test_coverage_distinguishes_from_iou(square_mask):
    """Une carte très large couvre toute la lésion (couverture 1) mais a une
    IoU faible. Ne pas confondre les deux."""
    everything = np.ones((100, 100), dtype=np.uint8)
    assert coverage(everything, square_mask) == 1.0
    assert iou_score(everything, square_mask) < 0.05


# ---------------------------------------------------------------------------
# Évaluation d'ensemble
# ---------------------------------------------------------------------------
def test_evaluate_skips_empty_masks(square_mask, centered_cam):
    cams = [centered_cam, centered_cam, centered_cam]
    masks = [square_mask, np.zeros((100, 100), dtype=np.uint8), square_mask]
    df = evaluate_localization(cams, masks)
    assert len(df) == 2, "les images sans masque doivent être ignorées"


def test_evaluate_all_empty_raises(centered_cam):
    with pytest.raises(ValueError, match="Aucune image évaluable"):
        evaluate_localization([centered_cam], [np.zeros((100, 100), dtype=np.uint8)])


def test_report_detects_localization_gap(square_mask, centered_cam):
    """Cas construit : le modèle vise juste quand il a raison, à côté sinon.
    Le rapport doit produire un écart positif marqué."""
    off = np.zeros((100, 100)); off[5, 5] = 1.0

    cams = [centered_cam] * 6 + [off] * 6
    masks = [square_mask] * 12
    labels = np.array([1] * 6 + [1] * 6)
    preds = np.array([1] * 6 + [0] * 6)          # 6 corrects, 6 erreurs

    rep = localization_report(evaluate_localization(cams, masks, labels, preds))
    assert rep["pointing_correct"] == 1.0
    assert rep["pointing_erreur"] == 0.0
    assert rep["pointing_gap"] == 1.0
    assert "participe bien à la décision" in rep["interpretation"]


def test_report_flags_suspicious_uniform_localization(square_mask, centered_cam):
    """Localisation identique que le modèle ait raison ou tort : le rapport
    doit signaler que la lésion ne participe peut-être pas à la décision."""
    cams = [centered_cam] * 12
    masks = [square_mask] * 12
    labels = np.array([1] * 6 + [0] * 6)
    preds = np.array([1] * 6 + [1] * 6)

    rep = localization_report(evaluate_localization(cams, masks, labels, preds))
    assert rep["pointing_gap"] == 0.0
    assert "indices hors lésion" in rep["interpretation"]


def test_report_has_required_keys(square_mask, centered_cam):
    df = evaluate_localization([centered_cam] * 4, [square_mask] * 4)
    rep = localization_report(df)
    for k in ("n_images", "iou_mean", "pointing_game", "coverage_mean"):
        assert k in rep
