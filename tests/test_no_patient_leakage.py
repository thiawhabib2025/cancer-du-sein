"""TEST CRITIQUE — RÈGLE N°1 : aucune patiente ne traverse deux splits.

Si ce test échoue, toutes les métriques du projet sont invalides.
Ne jamais le désactiver, ne jamais le marquer `xfail`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.splits import (
    PatientLeakageError,
    assert_no_patient_leakage,
    fold_split,
    make_folds,
    make_holdout,
    summarize_folds,
)


# ---------------------------------------------------------------------------
# Le détecteur détecte-t-il réellement quelque chose ?
# ---------------------------------------------------------------------------
def test_naive_image_split_is_caught(fake_manifest):
    """Un `train_test_split` naïf DOIT être rejeté.

    C'est le test le plus important du fichier : il prouve que le garde-fou
    n'est pas décoratif. Un découpage au niveau image sépare les vues CC et MLO
    d'une même patiente — exactement l'erreur que l'on veut rendre impossible.
    """
    train, test = train_test_split(fake_manifest, test_size=0.2, random_state=0)
    with pytest.raises(PatientLeakageError, match="FUITE PATIENTE"):
        assert_no_patient_leakage({"train": train, "test": test})


def test_detector_accepts_clean_split(fake_manifest):
    """Un découpage correct par patiente doit passer."""
    patients = fake_manifest["patient_id"].unique()
    train_p, test_p = patients[:150], patients[150:]
    splits = {
        "train": fake_manifest[fake_manifest["patient_id"].isin(train_p)],
        "test": fake_manifest[fake_manifest["patient_id"].isin(test_p)],
    }
    assert_no_patient_leakage(splits)  # ne doit pas lever


# ---------------------------------------------------------------------------
# Hold-out externe
# ---------------------------------------------------------------------------
def test_holdout_has_no_leakage(fake_manifest):
    dev, test = make_holdout(fake_manifest, test_size=0.15, seed=42)
    assert_no_patient_leakage({"dev": dev, "test": test})
    assert len(dev) + len(test) == len(fake_manifest), "des lignes ont été perdues"


def test_holdout_preserves_class_balance(fake_manifest):
    dev, test = make_holdout(fake_manifest, test_size=0.2, seed=42)
    assert abs(dev["label"].mean() - test["label"].mean()) < 0.15


def test_holdout_is_deterministic(fake_manifest):
    a, _ = make_holdout(fake_manifest, seed=7)
    b, _ = make_holdout(fake_manifest, seed=7)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Validation croisée
# ---------------------------------------------------------------------------
def test_folds_have_no_leakage(fake_manifest):
    folded = make_folds(fake_manifest, n_splits=5, seed=42)
    for fold in range(5):
        train, val = fold_split(folded, fold)
        assert_no_patient_leakage({f"train{fold}": train, f"val{fold}": val})


def test_every_row_gets_a_fold(fake_manifest):
    folded = make_folds(fake_manifest, n_splits=5, seed=42)
    assert (folded["fold"] >= 0).all()
    assert sorted(folded["fold"].unique()) == [0, 1, 2, 3, 4]


def test_all_images_of_a_patient_share_one_fold(fake_manifest):
    """Toutes les vues d'une patiente doivent tomber dans le même pli."""
    folded = make_folds(fake_manifest, n_splits=5, seed=42)
    n_folds_per_patient = folded.groupby("patient_id")["fold"].nunique()
    assert (n_folds_per_patient == 1).all(), (
        "Des patientes sont réparties sur plusieurs plis : "
        f"{n_folds_per_patient[n_folds_per_patient > 1].index.tolist()[:5]}"
    )


def test_folds_are_balanced(fake_manifest):
    """Le taux de positifs doit rester comparable d'un pli à l'autre."""
    folded = make_folds(fake_manifest, n_splits=5, seed=42)
    rates = folded.groupby("fold")["label"].mean()
    assert rates.max() - rates.min() < 0.20, f"stratification déséquilibrée : {rates.to_dict()}"


def test_folds_are_deterministic(fake_manifest):
    a = make_folds(fake_manifest, n_splits=5, seed=1)
    b = make_folds(fake_manifest, n_splits=5, seed=1)
    pd.testing.assert_series_equal(a["fold"], b["fold"])


# ---------------------------------------------------------------------------
# Chaîne complète
# ---------------------------------------------------------------------------
def test_full_pipeline_holdout_then_folds(fake_manifest):
    """Le scénario réel : hold-out d'abord, puis validation croisée sur le reste."""
    dev, test = make_holdout(fake_manifest, test_size=0.15, seed=42)
    dev = make_folds(dev, n_splits=5, seed=42)

    for fold in range(5):
        train, val = fold_split(dev, fold)
        assert_no_patient_leakage(
            {"train": train, "val": val, "test": test}
        )


def test_summarize_folds_reports_all(fake_manifest):
    folded = make_folds(fake_manifest, n_splits=5, seed=42)
    summary = summarize_folds(folded)
    assert len(summary) == 5
    assert summary["n_images"].sum() == len(fake_manifest)


# ---------------------------------------------------------------------------
# Cas dégradés
# ---------------------------------------------------------------------------
def test_too_few_patients_raises(fake_manifest):
    tiny = fake_manifest[fake_manifest["patient_id"].isin(
        fake_manifest["patient_id"].unique()[:3]
    )]
    with pytest.raises(ValueError, match="impossible"):
        make_folds(tiny, n_splits=5)


def test_missing_group_column_raises(fake_manifest):
    with pytest.raises(KeyError):
        make_folds(fake_manifest.drop(columns=["patient_id"]), n_splits=5)
