"""Tests des métriques et du test de DeLong.

Le test de DeLong est le fondement statistique de toutes les conclusions du
projet : c'est lui qui dira si la fusion bat réellement le mono-modal. Une
implémentation fausse invaliderait le rapport entier sans que rien ne le
signale — d'où cette batterie de contrôles contre des références indépendantes.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval.delong import auc_ci, delong_auc_variance, delong_test
from src.eval.metrics import (
    aggregate_folds,
    bootstrap_ci,
    compute_metrics,
    specificity_at_sensitivity,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.35, 500)
    return y, rng.normal(y * 1.2, 1.0), rng.normal(y * 0.9, 1.0)


# ---------------------------------------------------------------------------
# DeLong — cohérence avec une référence indépendante
# ---------------------------------------------------------------------------
def test_delong_auc_matches_sklearn(scores):
    """Contrôle le plus important : l'AUC de DeLong doit égaler celle de sklearn."""
    y, a, _ = scores
    auc, _ = delong_auc_variance(y, a)
    assert abs(auc - roc_auc_score(y, a)) < 1e-9


def test_delong_handles_ties(scores):
    """Les ex aequo sont fréquents (probabilités arrondies, modèles saturés)."""
    y, _, _ = scores
    rounded = np.round(np.random.default_rng(1).random(len(y)), 1)
    auc, var = delong_auc_variance(y, rounded)
    assert abs(auc - roc_auc_score(y, rounded)) < 1e-9
    assert var > 0


def test_identical_models_give_p_one(scores):
    """Un modèle comparé à lui-même : delta nul, p = 1."""
    y, a, _ = scores
    r = delong_test(y, a, a.copy())
    assert r["delta"] == 0.0
    assert r["p_value"] == 1.0
    assert not r["significant"]


def test_delong_is_antisymmetric(scores):
    """Inverser A et B doit inverser le signe du delta, pas la p-value."""
    y, a, b = scores
    r1 = delong_test(y, a, b)
    r2 = delong_test(y, b, a)
    assert abs(r1["delta"] + r2["delta"]) < 1e-9
    assert abs(r1["p_value"] - r2["p_value"]) < 1e-9


def test_variance_matches_hanley_mcneil(scores):
    """La variance doit rester du même ordre que l'approximation de Hanley-McNeil.

    Les deux ne coïncident pas exactement (Hanley-McNeil suppose des scores
    exponentiels), mais un écart d'un facteur 2 signalerait un bug.
    """
    y, a, _ = scores
    auc, var = delong_auc_variance(y, a)
    m, n = int(y.sum()), int((1 - y).sum())
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    hm = (auc * (1 - auc) + (m - 1) * (q1 - auc ** 2) + (n - 1) * (q2 - auc ** 2)) / (m * n)
    assert 0.7 < var / hm < 1.4


def test_large_difference_is_significant():
    """Un écart franc sur un grand échantillon doit ressortir significatif."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.4, 3000)
    strong = rng.normal(y * 2.0, 1.0)
    weak = rng.normal(y * 0.3, 1.0)
    r = delong_test(y, strong, weak)
    assert r["significant"]
    assert r["p_value"] < 0.001


def test_tiny_difference_is_not_significant():
    """Un écart minime doit rester non significatif — c'est le garde-fou contre
    les conclusions du type « 0,912 > 0,904 donc c'est mieux »."""
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.4, 200)
    a = rng.normal(y * 1.0, 1.0)
    b = a + rng.normal(0, 0.02, 200)
    assert not delong_test(y, a, b)["significant"]


def test_auc_ci_brackets_point_estimate(scores):
    y, a, _ = scores
    ci = auc_ci(y, a)
    assert ci["ci_low"] < ci["auc"] < ci["ci_high"]
    assert 0.0 <= ci["ci_low"] and ci["ci_high"] <= 1.0


def test_single_class_raises(scores):
    _, a, _ = scores
    with pytest.raises(ValueError, match="[Uu]ne seule classe"):
        delong_auc_variance(np.zeros(len(a)), a)


# ---------------------------------------------------------------------------
# Métriques cliniques
# ---------------------------------------------------------------------------
def test_perfect_classifier():
    y = np.array([0] * 50 + [1] * 50)
    m = compute_metrics(y, y.astype(float))
    assert m["auc_roc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["fn"] == 0


def test_specificity_at_sensitivity_respects_target(scores):
    """Le point de fonctionnement doit réellement atteindre la sensibilité visée."""
    y, a, _ = scores
    for target in (0.90, 0.95, 0.98):
        r = specificity_at_sensitivity(y, a, target)
        assert r["achieved"]
        assert r["sensitivity"] >= target - 1e-9


def test_higher_sensitivity_costs_specificity(scores):
    """Compromis fondamental du dépistage : plus de sensibilité, moins de spécificité."""
    y, a, _ = scores
    s90 = specificity_at_sensitivity(y, a, 0.90)["specificity"]
    s98 = specificity_at_sensitivity(y, a, 0.98)["specificity"]
    assert s90 >= s98


def test_metrics_include_all_cdc_requirements(scores):
    """Toutes les métriques du §6.5 du cahier des charges doivent être présentes."""
    y, a, _ = scores
    m = compute_metrics(y, a)
    for k in ("accuracy", "sensitivity", "specificity", "precision",
              "f1", "auc_roc", "mcc"):
        assert k in m, f"métrique '{k}' manquante"


def test_confusion_matrix_is_consistent(scores):
    y, a, _ = scores
    m = compute_metrics(y, a)
    assert m["tp"] + m["fn"] == m["n_positive"]
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == m["n"]


# ---------------------------------------------------------------------------
# Bootstrap groupé
# ---------------------------------------------------------------------------
def test_grouped_bootstrap_is_wider_than_ungrouped():
    """TEST CRITIQUE.

    Quand plusieurs images corrélées proviennent d'une même patiente, le
    bootstrap par image sous-estime l'incertitude. L'intervalle groupé doit
    être nettement plus large — sinon le rapport publierait des intervalles
    de confiance trop optimistes.
    """
    rng = np.random.default_rng(0)
    n_patients = 167
    y_pat = rng.binomial(1, 0.35, n_patients)
    latent = rng.normal(y_pat * 1.2, 1.0)

    y, s, g = [], [], []
    for p in range(n_patients):
        for _ in range(3):                       # 3 vues par patiente
            y.append(y_pat[p])
            g.append(p)
            s.append(latent[p] + rng.normal(0, 0.25))
    y, s, g = np.array(y), np.array(s), np.array(g)

    grouped = bootstrap_ci(y, s, groups=g, n_boot=500)
    ungrouped = bootstrap_ci(y, s, groups=None, n_boot=500)

    w_g = grouped["ci_high"] - grouped["ci_low"]
    w_u = ungrouped["ci_high"] - ungrouped["ci_low"]
    assert w_g > w_u * 1.2, (
        f"IC groupé ({w_g:.4f}) devrait dépasser l'IC par image ({w_u:.4f})"
    )
    assert grouped["grouped"] is True
    assert grouped["n_units"] == n_patients


def test_bootstrap_ci_contains_point_estimate(scores):
    y, a, _ = scores
    ci = bootstrap_ci(y, a, n_boot=200)
    assert ci["ci_low"] <= ci["value"] <= ci["ci_high"]


# ---------------------------------------------------------------------------
# Agrégation des plis
# ---------------------------------------------------------------------------
def test_aggregate_folds_reports_std():
    folds = [{"auc_roc": 0.90}, {"auc_roc": 0.92}, {"auc_roc": 0.88}]
    agg = aggregate_folds(folds)
    assert agg["auc_roc_mean"] == 0.9
    assert agg["auc_roc_std"] > 0
    assert agg["n_folds"] == 3


def test_aggregate_empty_is_safe():
    assert aggregate_folds([]) == {}
