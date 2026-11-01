"""
Validation **quantitative** de l'explicabilité — l'apport méthodologique du projet.

POURQUOI CE MODULE
------------------
Les travaux de l'état de l'art (Latha et al. 2024, Chen et al. 2025) présentent
des cartes Grad-CAM qui « paraissent correctes » et s'arrêtent là. C'est une
évaluation qualitative : on ne peut ni la chiffrer, ni la comparer, ni la
contredire.

Or BUSI, CBIS-DDSM, BrEaST et BUS-UCLM fournissent des **masques de lésion
annotés**. On peut donc mesurer
objectivement si le modèle regarde la lésion, et répondre à une question que
les radiologues posent réellement : *« le modèle regarde-t-il au bon endroit,
et dans quelle proportion des cas ? »*

DEUX MÉTRIQUES COMPLÉMENTAIRES
------------------------------
1. **IoU** entre carte seuillée et masque. Sévère : Grad-CAM produit des cartes
   grossières (activations sous-échantillonnées, typiquement 7×7 pour une
   entrée 224×224). Un IoU de 0,3 correspond déjà à une localisation correcte.

2. **Pointing game** : le maximum de la carte tombe-t-il dans la lésion ?
   Insensible à la finesse de la carte, c'est la métrique standard de la
   localisation faiblement supervisée, et la plus lisible pour un clinicien
   (« le modèle vise juste dans X % des cas »).

Rapporter les deux : l'IoU mesure le recouvrement, le pointing game la visée.
Une IoU faible avec un pointing game élevé signifie « bien centré, mal
délimité » — ce qui est le comportement attendu de Grad-CAM et non un défaut.

ANALYSE STRATIFIÉE
------------------
`localization_report` sépare les prédictions correctes des erreurs. Un modèle
qui localise bien quand il a raison **et** quand il a tort n'utilise pas la
lésion pour décider : c'est un signal d'alerte que la seule AUC ne révèle pas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "threshold_cam",
    "iou_score",
    "pointing_game",
    "coverage",
    "evaluate_localization",
    "localization_report",
]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def threshold_cam(cam: np.ndarray, method: str = "percentile", value: float = 80.0) -> np.ndarray:
    """Binarise une carte de chaleur.

    ``percentile`` (défaut) : conserve les `100 - value` % de pixels les plus
    activés. Robuste, car indépendant de l'échelle absolue des activations —
    contrairement à un seuil fixe, qui donnerait des résultats incomparables
    d'une image à l'autre.

    ``absolute`` : seuil fixe dans [0, 1], à n'utiliser que sur des cartes déjà
    normalisées de façon homogène.
    """
    cam = np.asarray(cam, dtype=np.float32)
    if method == "percentile":
        if not 0.0 <= value <= 100.0:
            raise ValueError("percentile attendu dans [0, 100]")
        thr = np.percentile(cam, value)
    elif method == "absolute":
        thr = value
    else:
        raise ValueError(f"method inconnue : {method}")
    return (cam > thr).astype(np.uint8)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection sur union de deux masques binaires."""
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"formes incompatibles : {p.shape} vs {g.shape}")
    union = np.logical_or(p, g).sum()
    if union == 0:
        return float("nan")            # aucun des deux masques n'a de pixel
    return float(np.logical_and(p, g).sum() / union)


def pointing_game(cam: np.ndarray, gt: np.ndarray) -> float:
    """1.0 si le maximum de la carte tombe dans la lésion, 0.0 sinon.

    NaN si le masque de référence est vide (rien à viser).
    """
    g = np.asarray(gt).astype(bool)
    if g.sum() == 0:
        return float("nan")
    cam = np.asarray(cam, dtype=np.float32)
    if cam.shape != g.shape:
        raise ValueError(f"formes incompatibles : {cam.shape} vs {g.shape}")
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return float(g[peak])


def coverage(pred: np.ndarray, gt: np.ndarray) -> float:
    """Proportion de la lésion couverte par la région saillante (rappel spatial).

    Complète l'IoU : une carte large couvrira bien la lésion (rappel élevé)
    tout en ayant une IoU faible. Distinguer les deux évite de conclure trop
    vite qu'un modèle « rate » la lésion.
    """
    g = np.asarray(gt).astype(bool)
    if g.sum() == 0:
        return float("nan")
    p = np.asarray(pred).astype(bool)
    return float(np.logical_and(p, g).sum() / g.sum())


# ---------------------------------------------------------------------------
# Évaluation sur un jeu
# ---------------------------------------------------------------------------
def evaluate_localization(
    cams: list[np.ndarray],
    masks: list[np.ndarray],
    labels: np.ndarray | None = None,
    preds: np.ndarray | None = None,
    percentile: float = 80.0,
) -> pd.DataFrame:
    """Calcule les métriques de localisation image par image.

    Parameters
    ----------
    cams, masks
        Listes de tableaux (H, W). Les masques sont binarisés (> 0).
    labels, preds
        Vérité et prédiction binaires, pour la stratification correct/erroné.

    Returns
    -------
    DataFrame : une ligne par image (``iou``, ``pointing``, ``coverage``,
    ``label``, ``pred``, ``correct``).
    """
    if len(cams) != len(masks):
        raise ValueError("cams et masks doivent avoir la même longueur")

    rows = []
    for i, (cam, mask) in enumerate(zip(cams, masks)):
        gt = np.asarray(mask) > 0
        if gt.sum() == 0:
            continue                    # image sans masque : non évaluable
        binary = threshold_cam(cam, "percentile", percentile)
        row = {
            "index": i,
            "iou": iou_score(binary, gt),
            "pointing": pointing_game(cam, gt),
            "coverage": coverage(binary, gt),
            "lesion_frac": float(gt.mean()),
        }
        if labels is not None:
            row["label"] = int(labels[i])
        if preds is not None:
            row["pred"] = int(preds[i])
            if labels is not None:
                row["correct"] = int(labels[i] == preds[i])
        rows.append(row)

    if not rows:
        raise ValueError("Aucune image évaluable : tous les masques sont vides.")
    return pd.DataFrame(rows)


def localization_report(df: pd.DataFrame) -> dict:
    """Synthèse des métriques, avec stratification par correction et par classe.

    Le chiffre à mettre en avant est ``pointing_game`` : « le modèle vise la
    lésion dans X % des cas ». L'IoU complète en quantifiant le recouvrement.
    """
    rep = {
        "n_images": int(len(df)),
        "iou_mean": round(float(df["iou"].mean()), 4),
        "iou_median": round(float(df["iou"].median()), 4),
        "pointing_game": round(float(df["pointing"].mean()), 4),
        "coverage_mean": round(float(df["coverage"].mean()), 4),
    }

    if "correct" in df.columns:
        for name, grp in [("correct", df[df.correct == 1]), ("erreur", df[df.correct == 0])]:
            if len(grp):
                rep[f"pointing_{name}"] = round(float(grp["pointing"].mean()), 4)
                rep[f"iou_{name}"] = round(float(grp["iou"].mean()), 4)
                rep[f"n_{name}"] = int(len(grp))

        # Écart de visée entre bonnes et mauvaises prédictions.
        if "pointing_correct" in rep and "pointing_erreur" in rep:
            gap = rep["pointing_correct"] - rep["pointing_erreur"]
            rep["pointing_gap"] = round(gap, 4)
            rep["interpretation"] = (
                "Le modèle vise nettement mieux quand il a raison : la lésion "
                "participe bien à la décision."
                if gap > 0.10 else
                "Écart faible entre bonnes et mauvaises prédictions : le modèle "
                "s'appuie peut-être sur des indices hors lésion (contexte, "
                "artefacts, annotations). À signaler dans le rapport."
            )

    if "label" in df.columns:
        for lab, name in [(1, "malin"), (0, "benin")]:
            grp = df[df.label == lab]
            if len(grp):
                rep[f"pointing_{name}"] = round(float(grp["pointing"].mean()), 4)

    return rep
