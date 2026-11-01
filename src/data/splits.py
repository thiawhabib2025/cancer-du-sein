"""
Découpage des données **par patiente**.

RÈGLE N°1 DU PROJET
-------------------
Aucun découpage ne se fait au niveau image. CBIS-DDSM contient plusieurs images
par patiente (CC, MLO, image pleine, ROI recadrée) ; BUSI contient plusieurs
images pour 600 patientes réparties sur 780 fichiers. Un `train_test_split`
naïf place la vue CC d'une patiente en entraînement et sa vue MLO en test :
le modèle mémorise la patiente au lieu d'apprendre la lésion.

Symptôme typique : AUC 0,97 en test interne, effondrement à 0,70 en externe.

Ce module est le **seul** point d'entrée autorisé pour créer des splits.
Toute fonction ici groupe sur `patient_id` et vérifie l'absence de fuite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    "PatientLeakageError",
    "assert_no_patient_leakage",
    "make_holdout",
    "make_folds",
]


class PatientLeakageError(AssertionError):
    """Une même patiente apparaît dans plusieurs splits."""


# ---------------------------------------------------------------------------
# Vérification
# ---------------------------------------------------------------------------
def assert_no_patient_leakage(
    splits: dict[str, pd.DataFrame],
    group_col: str = "patient_id",
) -> None:
    """Vérifie qu'aucune patiente n'apparaît dans deux splits différents.

    À appeler après **toute** création de split. C'est bon marché et cela
    protège la validité de l'ensemble des résultats du projet.

    Parameters
    ----------
    splits
        Dictionnaire ``{"train": df, "val": df, "test": df}``.
    group_col
        Colonne identifiant la patiente.

    Raises
    ------
    PatientLeakageError
        Dès qu'une intersection non vide est détectée.
    """
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a][group_col]) & set(splits[b][group_col])
            if overlap:
                sample = sorted(map(str, overlap))[:5]
                raise PatientLeakageError(
                    f"FUITE PATIENTE entre '{a}' et '{b}' : "
                    f"{len(overlap)} identifiant(s) partagé(s). Exemples : {sample}"
                )


def _check_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes absentes du manifeste : {missing}")


# ---------------------------------------------------------------------------
# Hold-out externe
# ---------------------------------------------------------------------------
def make_holdout(
    manifest: pd.DataFrame,
    test_size: float = 0.15,
    group_col: str = "patient_id",
    label_col: str = "label",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retire un jeu de test externe, groupé par patiente et stratifié par label.

    Ce hold-out est mis de côté **avant** la validation croisée et n'est utilisé
    qu'une seule fois, à la toute fin du projet. Le regarder plusieurs fois en
    ajustant le modèle revient à l'utiliser comme jeu de validation : les
    métriques finales deviennent optimistes.

    Returns
    -------
    (df_dev, df_test)
    """
    _check_columns(manifest, [group_col, label_col])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size doit être dans ]0, 1[, reçu {test_size}")

    # StratifiedGroupKFold avec n_splits = round(1/test_size) donne un pli
    # de taille ~test_size tout en respectant groupes et stratification.
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    dev_idx, test_idx = next(
        sgkf.split(manifest, y=manifest[label_col], groups=manifest[group_col])
    )
    df_dev = manifest.iloc[dev_idx].reset_index(drop=True)
    df_test = manifest.iloc[test_idx].reset_index(drop=True)

    assert_no_patient_leakage({"dev": df_dev, "test": df_test}, group_col)
    return df_dev, df_test


# ---------------------------------------------------------------------------
# Validation croisée
# ---------------------------------------------------------------------------
def make_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "patient_id",
    label_col: str = "label",
    seed: int = 42,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Assigne un numéro de pli à chaque ligne, en groupant par patiente.

    Toutes les images d'une même patiente reçoivent le même numéro de pli.

    Returns
    -------
    Copie du manifeste avec une colonne ``fold`` (entier de 0 à n_splits-1).
    """
    _check_columns(manifest, [group_col, label_col])

    n_patients = manifest[group_col].nunique()
    if n_patients < n_splits:
        raise ValueError(
            f"{n_patients} patiente(s) pour {n_splits} plis : impossible. "
            "Réduisez n_splits ou vérifiez le manifeste."
        )

    out = manifest.copy().reset_index(drop=True)
    out[fold_col] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        sgkf.split(out, y=out[label_col], groups=out[group_col])
    ):
        out.loc[val_idx, fold_col] = fold

    if (out[fold_col] == -1).any():
        raise RuntimeError("Certaines lignes n'ont pas reçu de pli — bug interne.")

    # Vérification systématique : chaque pli de validation est disjoint du reste.
    for fold in range(n_splits):
        assert_no_patient_leakage(
            {
                f"train_f{fold}": out[out[fold_col] != fold],
                f"val_f{fold}": out[out[fold_col] == fold],
            },
            group_col,
        )
    return out


def fold_split(
    manifest: pd.DataFrame, fold: int, fold_col: str = "fold"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne ``(train, val)`` pour un pli donné."""
    if fold_col not in manifest.columns:
        raise KeyError(f"'{fold_col}' absent — exécutez make_folds() d'abord.")
    train = manifest[manifest[fold_col] != fold].reset_index(drop=True)
    val = manifest[manifest[fold_col] == fold].reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
def summarize_folds(
    manifest: pd.DataFrame,
    fold_col: str = "fold",
    label_col: str = "label",
    group_col: str = "patient_id",
) -> pd.DataFrame:
    """Tableau récapitulatif par pli — à inspecter avant tout entraînement.

    Un déséquilibre marqué du taux de positifs entre plis signale un problème
    de stratification (souvent trop peu de patientes par rapport à n_splits).
    """
    rows = []
    for fold, grp in manifest.groupby(fold_col):
        rows.append(
            {
                "fold": fold,
                "n_images": len(grp),
                "n_patients": grp[group_col].nunique(),
                "n_positive": int(grp[label_col].sum()),
                "pos_rate": round(float(grp[label_col].mean()), 4),
                "datasets": ", ".join(sorted(grp["dataset"].unique()))
                if "dataset" in grp.columns
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Découpage par patiente du manifeste.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Manifeste : {len(df)} images · {df['patient_id'].nunique()} patientes")

    df_dev, df_test = make_holdout(df, args.test_size, seed=args.seed)
    df_dev = make_folds(df_dev, args.n_splits, seed=args.seed)
    df_test["fold"] = -1  # convention : -1 = hold-out externe

    out = pd.concat([df_dev, df_test], ignore_index=True)
    out_path = args.out or args.manifest
    out.to_csv(out_path, index=False)

    print("\nRépartition des plis (fold = -1 → hold-out externe) :")
    print(summarize_folds(out).to_string(index=False))
    print(f"\n✅ Aucune fuite patiente détectée. Écrit dans {out_path}")


if __name__ == "__main__":
    main()
