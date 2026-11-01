"""TEST CRITIQUE — RÈGLE N°2 : les variables cliniques synthétiques ne doivent
pas encoder le label.

Plafond : AUC ≤ 0,65 sur les variables cliniques seules — l'ordre de grandeur
des modèles de risque réels (Gail : 0,58–0,62).

Si ce test échoue, le modèle « multi-modal » apprendra surtout à inverser le
générateur de données cliniques, et les AUC rapportées seront fictives.
"""

from __future__ import annotations

import pytest

from src.data.clinical_synth import (
    ClinicalLeakageError,
    ClinicalPriors,
    audit_leakage,
    generate,
)


# ---------------------------------------------------------------------------
# Le générateur par défaut respecte-t-il le plafond ?
# ---------------------------------------------------------------------------
def test_default_priors_stay_under_cap(fake_manifest):
    """Avec les tailles d'effet publiées, l'AUC clinique doit rester ≤ 0,65."""
    df = generate(fake_manifest, seed=42)
    report = audit_leakage(df, strict=True, seed=42)
    assert report["passed"]
    assert report["auc_clinical_only"] <= 0.65


def test_audit_reports_per_variable_auc(fake_manifest):
    """L'audit doit dire QUELLE variable contribue le plus — sinon il est inutile
    pour déboguer un générateur qui fuit."""
    df = generate(fake_manifest, seed=42)
    report = audit_leakage(df, strict=False, seed=42)
    assert report["per_variable_auc"], "aucune contribution par variable rapportée"
    assert all(0.0 <= v <= 1.0 for v in report["per_variable_auc"].values())


# ---------------------------------------------------------------------------
# Le garde-fou détecte-t-il réellement une fuite ?
# ---------------------------------------------------------------------------
def test_exaggerated_effects_are_caught(fake_manifest):
    """Test le plus important du fichier.

    On gonfle délibérément les tailles d'effet — ce que ferait naturellement
    quelqu'un cherchant à rendre les données « plus séparables ». Le garde-fou
    DOIT le refuser. S'il ne le fait pas, il ne protège rien.
    """
    leaky = ClinicalPriors(
        family_history_prev_control=0.05,
        family_history_or=25.0,      # ← absurde : OR réel ≈ 2
        brca_prev_control=0.001,
        brca_prev_case=0.60,         # ← absurde : ≈ 5 % en réalité
        bmi_shift_case=12.0,         # ← absurde : ≈ 1 kg/m² en réalité
        density_or_dense=20.0,
    )
    df = generate(fake_manifest, priors=leaky, seed=42)

    with pytest.raises(ClinicalLeakageError, match="RÈGLE N°2|plafond"):
        audit_leakage(df, strict=True, seed=42)


def test_non_strict_mode_reports_without_raising(fake_manifest):
    """En mode diagnostic, on veut le rapport sans interrompre l'exécution."""
    leaky = ClinicalPriors(family_history_or=25.0, bmi_shift_case=12.0)
    df = generate(fake_manifest, priors=leaky, seed=42)
    report = audit_leakage(df, strict=False, seed=42)
    assert report["passed"] is False


# ---------------------------------------------------------------------------
# Cohérence de la génération
# ---------------------------------------------------------------------------
def test_one_value_per_patient(fake_manifest):
    """Une patiente a un seul IMC, quel que soit son nombre d'images.

    Générer par ligne donnerait à la même femme plusieurs IMC — incohérent
    cliniquement, et exploitable par le modèle comme signal parasite.
    """
    df = generate(fake_manifest, seed=42)
    for col in ("bmi", "family_history", "brca", "density_synth"):
        n_unique = df.groupby("patient_id")[col].nunique()
        assert (n_unique == 1).all(), f"'{col}' varie au sein d'une même patiente"


def test_generation_is_deterministic(fake_manifest):
    a = generate(fake_manifest, seed=123)
    b = generate(fake_manifest, seed=123)
    assert a["bmi"].equals(b["bmi"])
    assert a["family_history"].equals(b["family_history"])


def test_values_are_physiologically_plausible(fake_manifest):
    df = generate(fake_manifest, seed=42)
    assert df["bmi"].between(15, 50).all(), "IMC hors bornes physiologiques"
    assert df["family_history"].isin([0, 1]).all()
    assert df["brca"].isin([0, 1]).all()
    assert df["density_synth"].isin(["A", "B", "C", "D"]).all()


def test_effect_directions_match_epidemiology(fake_manifest):
    """Les effets doivent aller dans le bon sens, même s'ils sont faibles.

    Un générateur sans aucun effet passerait le plafond trivialement mais ne
    simulerait rien de réaliste. On vérifie donc que le signal existe.
    """
    df = generate(fake_manifest, seed=42)
    per_patient = df.groupby("patient_id").agg({"label": "max", "family_history": "first"})
    rate_case = per_patient.loc[per_patient["label"] == 1, "family_history"].mean()
    rate_ctrl = per_patient.loc[per_patient["label"] == 0, "family_history"].mean()
    assert rate_case > rate_ctrl, "les antécédents devraient être plus fréquents chez les cas"


def test_age_synthesis_is_opt_in(fake_manifest):
    """L'âge ne doit pas être généré par défaut : c'est le contributeur de
    fuite le plus puissant, et CMMD fournit l'âge réel."""
    df = generate(fake_manifest, seed=42)
    assert "age_synth" not in df.columns
