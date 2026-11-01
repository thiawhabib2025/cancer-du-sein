"""
Test de DeLong — comparaison statistique de deux AUC sur les **mêmes** patientes.

POURQUOI CE TEST EST INDISPENSABLE ICI
--------------------------------------
Le projet compare une dizaine de configurations : 5 architectures, 4 stratégies
de fusion, mono-modal contre multi-modal. Sans test statistique, la conclusion
se réduit à « 0,912 > 0,904, donc la fusion est meilleure » — ce qui n'a aucune
valeur quand l'écart-type entre plis dépasse l'écart observé.

DeLong (1988), dans l'implémentation rapide de Sun & Xu (2014), teste si deux
AUC calculées **sur le même jeu de test** diffèrent significativement. Le point
crucial est qu'il tient compte de la corrélation entre les deux modèles : deux
modèles évalués sur les mêmes patientes se trompent souvent sur les mêmes cas,
ce qu'un test non apparié ignorerait, produisant des p-values trop
conservatrices.

C'est le test que réclame le cahier des charges (§6.5) et qu'utilisent
Sharma et al. (2025).

Référence
---------
Sun X., Xu W. (2014). *Fast Implementation of DeLong's Algorithm for Comparing
the Areas Under Correlated Receiver Operating Characteristic Curves.*
IEEE Signal Processing Letters, 21(11), 1389-1393.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["delong_auc_variance", "delong_test", "auc_ci"]


# ---------------------------------------------------------------------------
# Rangs à moyennes égales (midrank)
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Rangs avec moyenne sur les ex aequo.

    Les ex aequo sont fréquents avec des probabilités arrondies ou des modèles
    très confiants ; les traiter naïvement biaise la variance.
    """
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1  # rang moyen, base 1
        i = j + 1

    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _structural_components(
    predictions: np.ndarray, n_pos: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composantes structurelles V10 / V01 de DeLong.

    `predictions` doit être de forme (n_modeles, n_echantillons), les `n_pos`
    premières colonnes correspondant aux positifs.
    """
    m, n = n_pos, predictions.shape[1] - n_pos
    pos = predictions[:, :m]
    neg = predictions[:, m:]
    k = predictions.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(predictions[r])

    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v01, v10


def _covariance(v01: np.ndarray, v10: np.ndarray, m: int, n: int) -> np.ndarray:
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    # np.cov renvoie un scalaire 0-d pour un seul modèle : on reforme en 2D.
    s01 = np.atleast_2d(s01)
    s10 = np.atleast_2d(s10)
    return s01 / m + s10 / n


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def _prepare(y_true: np.ndarray, *scores: np.ndarray):
    y_true = np.asarray(y_true).ravel()
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true doit être binaire (0/1).")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Une seule classe présente — AUC indéfinie.")

    order = np.argsort(-y_true, kind="mergesort")  # positifs d'abord, stable
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())

    mat = np.vstack([np.asarray(s).ravel()[order] for s in scores])
    return mat, n_pos, len(y_true) - n_pos


def delong_auc_variance(y_true, y_score) -> tuple[float, float]:
    """Retourne ``(auc, variance)`` pour un modèle.

    L'AUC renvoyée doit coïncider avec ``sklearn.metrics.roc_auc_score`` —
    c'est le contrôle de cohérence le plus simple de l'implémentation.
    """
    mat, m, n = _prepare(y_true, y_score)
    auc, v01, v10 = _structural_components(mat, m)
    cov = _covariance(v01, v10, m, n)
    return float(auc[0]), float(cov[0, 0])


def auc_ci(y_true, y_score, alpha: float = 0.05) -> dict:
    """Intervalle de confiance analytique de l'AUC (DeLong).

    Plus rapide que le bootstrap et sans tirage aléatoire, donc reproductible.
    ⚠️ Suppose des observations indépendantes : à ne pas utiliser tel quel
    quand plusieurs images proviennent de la même patiente — voir
    ``metrics.bootstrap_ci`` qui gère le groupement.
    """
    auc, var = delong_auc_variance(y_true, y_score)
    se = float(np.sqrt(max(var, 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc": round(auc, 4),
        "se": round(se, 4),
        "ci_low": round(max(0.0, auc - z * se), 4),
        "ci_high": round(min(1.0, auc + z * se), 4),
        "alpha": alpha,
    }


def delong_test(y_true, score_a, score_b, name_a: str = "A", name_b: str = "B") -> dict:
    """Compare deux AUC évaluées sur les mêmes échantillons.

    Hypothèse nulle : AUC(A) = AUC(B).

    Returns
    -------
    dict avec ``auc_a``, ``auc_b``, ``delta``, ``z``, ``p_value``,
    ``significant`` (au seuil 0,05) et l'IC à 95 % de la différence.

    Notes
    -----
    Un p non significatif est une information, pas un échec. « La cross-attention
    n'améliore pas significativement la moyenne des logits » est un résultat
    publiable — et c'est précisément ce que le protocole de la piste B cherche
    à établir (00_PLAN_PROJET.md, Décision n°1).
    """
    mat, m, n = _prepare(y_true, score_a, score_b)
    aucs, v01, v10 = _structural_components(mat, m)
    cov = _covariance(v01, v10, m, n)

    delta = float(aucs[0] - aucs[1])
    var_delta = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var_delta, 0.0)))

    if se < 1e-12:
        z, p = 0.0, 1.0     # modèles identiques
    else:
        z = delta / se
        p = float(2 * (1 - stats.norm.cdf(abs(z))))

    ci = stats.norm.ppf(0.975) * se
    return {
        "name_a": name_a,
        "name_b": name_b,
        "auc_a": round(float(aucs[0]), 4),
        "auc_b": round(float(aucs[1]), 4),
        "delta": round(delta, 4),
        "delta_ci_low": round(delta - ci, 4),
        "delta_ci_high": round(delta + ci, 4),
        "se": round(se, 5),
        "z": round(z, 4),
        "p_value": round(p, 6),
        "significant": bool(p < 0.05),
    }


def pairwise_delong(y_true, scores: dict[str, np.ndarray]) -> "pd.DataFrame":
    """Matrice de comparaisons deux à deux entre toutes les configurations.

    À produire à la fin des Phases 2 et 3 : c'est la table qui justifie le choix
    d'architecture et la conclusion sur la fusion.
    """
    import pandas as pd
    from itertools import combinations

    rows = [
        delong_test(y_true, scores[a], scores[b], a, b)
        for a, b in combinations(scores, 2)
    ]
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
