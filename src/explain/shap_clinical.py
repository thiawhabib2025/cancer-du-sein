"""
SHAP sur les variables cliniques **réelles** (niveau C0).

Contexte : conformément à la décision d'équipe, seules l'âge, la densité
BI-RADS et le type de lésion entrent dans le modèle. L'analyse SHAP répond à
la question du radiologue : *« qu'est-ce qui, dans le dossier clinique, a fait
pencher la décision ? »*

CE QU'IL FAUT ATTENDRE — ET NE PAS SUR-INTERPRÉTER
--------------------------------------------------
Avec trois variables, la contribution clinique sera **modeste** : les modèles
de risque cliniques établis plafonnent autour de 0,60 d'AUC. Un modèle
clinique seul qui dépasserait 0,65 signalerait une fuite d'information, pas une
découverte.

L'intérêt de SHAP ici n'est donc pas de démontrer que la clinique prédit le
cancer, mais de **rendre lisible** la part qu'elle prend dans la décision
finale — et de vérifier que le modèle utilise ces variables dans le sens
attendu (le risque croît avec l'âge, avec la densité mammaire).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["clinical_shap_values", "shap_summary_table", "plot_shap_summary"]


def clinical_shap_values(
    model,
    background: np.ndarray,
    samples: np.ndarray,
    feature_names: list[str],
    device: str = "cpu",
    nsamples: int = 200,
):
    """Valeurs de Shapley pour les variables cliniques d'un modèle PyTorch.

    Utilise ``KernelExplainer`` plutôt que ``DeepExplainer`` : il est agnostique
    à l'architecture, donc utilisable aussi bien sur le MLP clinique seul que
    sur un modèle de fusion dont on fige la partie image.

    Parameters
    ----------
    model
        Modèle renvoyant un logit à partir d'un vecteur clinique.
    background
        Échantillon de référence (50 à 100 lignes suffisent ; au-delà, le coût
        explose sans gain notable).
    samples
        Lignes à expliquer.

    Returns
    -------
    np.ndarray de forme (n_samples, n_features).
    """
    import shap
    import torch

    model.eval().to(device)

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(x, dtype=np.float32), device=device)
            out = model(t)
            return torch.sigmoid(out).cpu().numpy().ravel()

    # k-means sur le fond : réduit fortement le coût de KernelExplainer,
    # qui est quadratique en nombre de références.
    bg = shap.kmeans(background, min(25, len(background)))
    explainer = shap.KernelExplainer(predict, bg)
    values = explainer.shap_values(samples, nsamples=nsamples, silent=True)

    values = np.asarray(values)
    if values.ndim == 3:                 # certaines versions renvoient (n, f, 1)
        values = values[..., 0]
    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"{values.shape[1]} colonnes SHAP pour {len(feature_names)} variables — "
            "vérifiez l'ordre de ClinicalEncoder.transform_row()."
        )
    return values


def shap_summary_table(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Classement des variables par importance moyenne (|SHAP|).

    ``effet_moyen`` conserve le signe : positif = pousse vers « malin ».
    C'est cette colonne qui permet le contrôle de cohérence clinique — un âge
    élevé doit pousser vers le malin, pas l'inverse.
    """
    values = np.asarray(values)
    df = pd.DataFrame({
        "variable": feature_names,
        "importance": np.abs(values).mean(axis=0),
        "effet_moyen": values.mean(axis=0),
        "ecart_type": values.std(axis=0),
    })
    df["importance_rel"] = df["importance"] / (df["importance"].sum() + 1e-12)
    return df.sort_values("importance", ascending=False).round(4).reset_index(drop=True)


def plot_shap_summary(values: np.ndarray, feature_names: list[str], top_n: int = 10):
    """Diagramme d'importance — figure prête pour le rapport."""
    import matplotlib.pyplot as plt

    tbl = shap_summary_table(values, feature_names).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(tbl) + 1.5))
    colors = ["#c0392b" if e > 0 else "#2c6fbb" for e in tbl["effet_moyen"]]
    ax.barh(tbl["variable"], tbl["importance"], color=colors)
    ax.set_xlabel("importance moyenne |SHAP|")
    ax.set_title("Contribution des variables cliniques\n"
                 "(rouge : pousse vers malin · bleu : vers bénin)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    return fig
