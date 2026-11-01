"""
Métriques d'évaluation — orientées usage clinique.

DEUX PARTIS PRIS QUI COMPTENT
-----------------------------

**1. Le seuil 0,5 n'a aucun sens clinique.** La quasi-totalité des articles
rapporte sensibilité et spécificité au seuil 0,5, hérité du softmax. Or en
dépistage, on ne choisit pas ce seuil : on fixe d'abord la sensibilité qu'on
s'impose — « ne pas manquer plus de 5 % des cancers » — puis on regarde la
spécificité obtenue à ce point de fonctionnement.

C'est le sens de `specificity_at_sensitivity()`. La question clinique n'est pas
« quelle est la précision du modèle ? » mais **« combien de biopsies inutiles
évite-t-on à sensibilité fixée ? »**. C'est aussi la métrique qui parlera
directement aux trois radiologues de la validation.

**2. Les intervalles de confiance doivent être groupés par patiente.** Un
bootstrap qui rééchantillonne les images traite les 3 vues d'une même femme
comme 3 observations indépendantes. Il sous-estime la variance et produit des
intervalles trop étroits — le même défaut que le découpage par image, en plus
discret. `bootstrap_ci()` rééchantillonne les **patientes**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

__all__ = [
    "compute_metrics",
    "specificity_at_sensitivity",
    "best_threshold",
    "bootstrap_ci",
    "aggregate_patient_scores",
]


# ---------------------------------------------------------------------------
# Point de fonctionnement
# ---------------------------------------------------------------------------
def specificity_at_sensitivity(
    y_true, y_score, target_sensitivity: float = 0.95
) -> dict:
    """Spécificité obtenue au seuil garantissant la sensibilité visée.

    C'est LA métrique à mettre en avant dans le rapport et devant les
    radiologues : « à 95 % de sensibilité, le modèle évite X % des biopsies
    inutiles ». Elle traduit directement les deux objectifs du §10.1 du cahier
    des charges — moins de cancers manqués, moins de biopsies inutiles.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        return {
            "target_sensitivity": target_sensitivity,
            "achieved": False,
            "threshold": None,
            "specificity": 0.0,
            "sensitivity": float(tpr.max()),
        }

    i = valid[0]        # premier point atteignant la cible = meilleure spécificité
    return {
        "target_sensitivity": target_sensitivity,
        "achieved": True,
        "threshold": round(float(thresholds[i]), 4),
        "specificity": round(float(1 - fpr[i]), 4),
        "sensitivity": round(float(tpr[i]), 4),
    }


def best_threshold(y_true, y_score, criterion: str = "youden") -> float:
    """Seuil optimal.

    ``youden``  maximise sensibilité + spécificité − 1 (équilibré)
    ``f1``      maximise le F1 (utile en fort déséquilibre de classes)

    ⚠️ Le seuil DOIT être choisi sur le split de validation, jamais sur le test :
    l'optimiser sur le test revient à y ajuster le modèle.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_score)
        return float(thr[np.argmax(tpr - fpr)])

    if criterion == "f1":
        candidates = np.unique(y_score)
        if len(candidates) > 500:
            candidates = np.quantile(y_score, np.linspace(0, 1, 500))
        scores = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
                  for t in candidates]
        return float(candidates[int(np.argmax(scores))])

    raise ValueError(f"criterion inconnu : {criterion}")


# ---------------------------------------------------------------------------
# Bilan complet
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true,
    y_score,
    threshold: float | None = None,
    target_sensitivities: tuple[float, ...] = (0.90, 0.95, 0.98),
) -> dict:
    """Toutes les métriques du §6.5 du cahier des charges, plus les points de
    fonctionnement cliniques.

    Parameters
    ----------
    threshold
        Si ``None``, le seuil de Youden est utilisé. En évaluation finale,
        passez le seuil déterminé **sur la validation**.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel()

    if len(np.unique(y_true)) < 2:
        raise ValueError("Une seule classe présente — métriques indéfinies.")

    thr = best_threshold(y_true, y_score) if threshold is None else threshold
    y_pred = (y_score >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "auc_roc": round(float(roc_auc_score(y_true, y_score)), 4),
        "auc_pr": round(float(average_precision_score(y_true, y_score)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # Sensibilité = rappel. CRITIQUE : c'est le taux de cancers détectés.
        "sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "threshold": round(float(thr), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": round(float(y_true.mean()), 4),
    }

    # Points de fonctionnement cliniques
    for s in target_sensitivities:
        r = specificity_at_sensitivity(y_true, y_score, s)
        out[f"spec_at_sens{int(s * 100)}"] = r["specificity"]

    return out


# ---------------------------------------------------------------------------
# Incertitude
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true,
    y_score,
    groups=None,
    metric: str = "auc_roc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Intervalle de confiance par bootstrap, **groupé par patiente**.

    Quand `groups` est fourni (typiquement `patient_id`), le rééchantillonnage
    porte sur les patientes et non sur les images. Sans cela, les 3 vues d'une
    même femme comptent comme 3 observations indépendantes : la variance est
    sous-estimée et l'intervalle trop étroit.

    C'est la version rigoureuse à utiliser dans le rapport. `delong.auc_ci`
    est plus rapide mais suppose l'indépendance.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel()
    rng = np.random.default_rng(seed)

    is_grouped = groups is not None
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups).ravel()

    unique_groups = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

    stats_boot: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue                     # tirage dégénéré : une seule classe
        try:
            stats_boot.append(compute_metrics(y_true[idx], y_score[idx])[metric])
        except ValueError:
            continue

    if not stats_boot:
        raise RuntimeError("Tous les tirages bootstrap étaient dégénérés.")

    arr = np.asarray(stats_boot)
    point = compute_metrics(y_true, y_score)[metric]
    return {
        "metric": metric,
        "value": point,
        "ci_low": round(float(np.percentile(arr, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 4),
        "std": round(float(arr.std()), 4),
        "n_boot": len(arr),
        "grouped": is_grouped,
        "n_units": int(len(unique_groups)),
    }


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Agrège les métriques de plusieurs plis : moyenne ± écart-type.

    L'écart-type entre plis est ce qui permet de juger si un écart de 0,008
    d'AUC entre deux architectures signifie quoi que ce soit.
    """
    if not fold_metrics:
        return {}
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.asarray([m[k] for m in fold_metrics], dtype=float)
        out[f"{k}_mean"] = round(float(vals.mean()), 4)
        out[f"{k}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
    out["n_folds"] = len(fold_metrics)
    return out


def aggregate_patient_scores(
    y_true,
    y_score,
    patient_ids,
    score_aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Agrège les prédictions image en une observation par patiente.

    Pour une patiente portant plusieurs labels (par exemple des seins de
    classes différentes), le label agrégé vaut 1 si au moins une image est
    maligne. Les scores sont moyennés par défaut. Les identifiants sont triés
    pour garantir un ordre commun lors des comparaisons entre modèles.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    patient_ids = np.asarray(patient_ids).ravel()

    if not (len(y_true) == len(y_score) == len(patient_ids)):
        raise ValueError("y_true, y_score et patient_ids doivent avoir la même longueur.")
    if len(y_true) == 0:
        raise ValueError("Impossible d'agréger un jeu vide.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true doit être binaire (0/1).")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contient des valeurs non finies.")
    if any(pd.isna(patient_id) for patient_id in patient_ids):
        raise ValueError("patient_ids ne peut pas contenir de valeurs manquantes.")
    if score_aggregation not in {"mean", "max"}:
        raise ValueError("score_aggregation doit être 'mean' ou 'max'.")

    unique_ids = np.unique(patient_ids)
    patient_labels = np.empty(len(unique_ids), dtype=int)
    patient_scores = np.empty(len(unique_ids), dtype=float)
    for i, patient_id in enumerate(unique_ids):
        selected = patient_ids == patient_id
        scores = y_score[selected]
        patient_labels[i] = int(np.max(y_true[selected]))
        patient_scores[i] = float(
            np.mean(scores) if score_aggregation == "mean" else np.max(scores)
        )

    return patient_labels, patient_scores, unique_ids
