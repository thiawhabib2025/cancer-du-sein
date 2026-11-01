"""
Génération des variables cliniques synthétiques (niveau **C2**) — avec garde-fou anti-fuite.

RÈGLE N°2 DU PROJET
-------------------
Les datasets publics d'imagerie ne contiennent pas d'antécédents familiaux, de
statut BRCA ni d'IMC. Le cahier des charges prévoit de les générer. C'est
possible, mais le piège est mortel :

    Si l'on génère ces variables en conditionnant fortement sur le label,
    le modèle n'apprend plus la pathologie — il apprend à INVERSER LE GÉNÉRATEUR.

Le symptôme est reconnaissable : MLP clinique seul à 0,95 d'AUC, fusion à 0,98,
et des chiffres qui ne survivent pas à la première question du jury.

Le garde-fou est simple et non négociable :

    Un classifieur entraîné sur les seules variables synthétiques
    doit rester SOUS 0,65 d'AUC.

C'est l'ordre de grandeur des modèles de risque clinique réels : le modèle de
Gail, référence du domaine, plafonne autour de 0,58–0,62 en discrimination
individuelle. Un modèle qui ferait mieux à partir de trois variables de risque
n'est pas un bon modèle — c'est un générateur qui fuit.

Ce module :
  1. génère les variables à partir de **prévalences et odds ratios publiés**,
     et non de règles ad hoc ;
  2. **auto-audite** systématiquement le résultat et refuse d'écrire un jeu
     de données qui dépasse le plafond.

Niveaux de données cliniques (voir 00_PLAN_PROJET.md, Décision n°2)
    C0  réel        — âge (CMMD, BrEaST), densité BI-RADS (INbreast), type de lésion
    C1  dérivé      — densité estimée depuis l'image
    C2  synthétique — antécédents familiaux, BRCA, IMC   ← CE MODULE

Les variables C2 ne figurent JAMAIS dans les résultats principaux du rapport.
Elles servent uniquement d'étude d'ablation explicitement étiquetée.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["ClinicalPriors", "ClinicalLeakageError", "generate", "audit_leakage"]

DEFAULT_MAX_AUC = 0.65


class ClinicalLeakageError(AssertionError):
    """Les variables cliniques synthétiques encodent trop fortement le label."""


# ---------------------------------------------------------------------------
# Paramètres épidémiologiques
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClinicalPriors:
    """Prévalences et tailles d'effet issues de la littérature épidémiologique.

    Chaque valeur doit rester traçable jusqu'à une source publiée : c'est ce qui
    distingue une simulation défendable d'un jeu de nombres inventés. Vérifiez
    ces valeurs avec un clinicien et consignez la source exacte dans le rapport.

    Sources indicatives (à confirmer et citer précisément dans le mémoire) :
      - Antécédents familiaux au 1er degré : ~13 % en population témoin,
        risque relatif ~2,0 (Collaborative Group on Hormonal Factors in
        Breast Cancer, Lancet 2001).
      - BRCA1/2 : ~0,25 % en population générale ; ~5 % des cancers du sein
        sont porteurs d'une mutation germinale (NCCN / littérature usuelle).
      - IMC : association modeste en post-ménopause, OR ~1,3 par 5 kg/m².
      - Densité mammaire : distribution BI-RADS ~10/40/40/10 % pour A/B/C/D ;
        seins denses (C/D) associés à un risque ~2× (Boyd et al., NEJM 2007).

    ⚠️ Les tailles d'effet doivent rester MODESTES. C'est précisément parce que
    les facteurs de risque cliniques discriminent mal au niveau individuel que
    l'imagerie reste indispensable. Gonfler ces effets produit une fuite.

    ⚠️ Point méthodologique important. Les odds ratios publiés décrivent le
    risque de *développer* un cancer dans une population générale. Or notre
    cohorte est diagnostique : toutes les femmes du dataset ont déjà été
    imagées, et beaucoup biopsiées. Le contraste cas/témoins y est nettement
    plus faible que dans une étude de dépistage — les témoins sont eux aussi
    des femmes symptomatiques ou à lésion suspecte. Les valeurs ci-dessous sont
    donc **délibérément atténuées** par rapport aux OR de la littérature de
    dépistage. C'est un choix conservateur et défendable : sous-estimer l'effet
    ne peut que sous-estimer la contribution clinique, jamais la gonfler.
    """

    # Antécédents familiaux au premier degré (binaire)
    family_history_prev_control: float = 0.13
    family_history_or: float = 1.8       # littérature dépistage ~2,0 — atténué

    # Mutation BRCA1/2 (binaire) — très rare, donc faible pouvoir discriminant
    brca_prev_control: float = 0.0025
    brca_prev_case: float = 0.05

    # IMC (continu, kg/m²)
    bmi_mean_control: float = 26.0
    bmi_sd: float = 5.0
    bmi_shift_case: float = 0.8          # ≈ OR 1,2 par 5 kg/m² — atténué

    # Âge (continu) — généré uniquement si absent des métadonnées réelles
    age_mean_control: float = 52.0
    age_sd: float = 12.0
    age_shift_case: float = 5.0
    age_min: float = 20.0
    age_max: float = 90.0

    # Densité BI-RADS A/B/C/D (catégoriel)
    density_probs_control: tuple[float, ...] = (0.10, 0.40, 0.40, 0.10)
    density_or_dense: float = 1.3        # C ou D vs A ou B — littérature ~2,0, atténué
                                         # (c'était la principale source de fuite
                                         #  détectée lors de la calibration initiale)


DENSITY_LEVELS = ("A", "B", "C", "D")


def _prob_from_or(p_control: float, odds_ratio: float) -> float:
    """Convertit une prévalence témoin + un odds ratio en prévalence chez les cas."""
    odds_control = p_control / (1.0 - p_control)
    odds_case = odds_control * odds_ratio
    return float(odds_case / (1.0 + odds_case))


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------
def generate(
    manifest: pd.DataFrame,
    priors: ClinicalPriors | None = None,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "patient_id",
    fill_missing_age: bool = False,
) -> pd.DataFrame:
    """Ajoute les variables cliniques synthétiques au manifeste.

    Points de conception importants
    -------------------------------
    1. **Génération au niveau patiente, pas image.** Une patiente a un seul IMC.
       Générer par ligne donnerait à la même femme trois IMC différents selon la
       vue — incohérent, et cela fournirait au modèle un signal de type
       « moyenner plusieurs tirages » exploitable.
    2. **Effets modestes**, issus de la littérature (voir `ClinicalPriors`).
    3. **Seed fixe**, génération faite AVANT tout découpage, pour que les mêmes
       patientes reçoivent toujours les mêmes valeurs d'une exécution à l'autre.

    Parameters
    ----------
    fill_missing_age
        Si ``True``, complète les âges manquants. À utiliser avec prudence :
        l'âge est le facteur de risque le plus discriminant, c'est donc le plus
        gros contributeur potentiel de fuite. Préférez laisser ``NaN`` et
        utiliser CMMD, qui fournit l'âge réel.

    Returns
    -------
    Copie du manifeste enrichie de ``family_history``, ``brca``, ``bmi``,
    ``density_synth`` et ``clinical_level``.
    """
    priors = priors or ClinicalPriors()
    rng = np.random.default_rng(seed)
    df = manifest.copy()

    for col in (label_col, group_col):
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' absente du manifeste.")

    # --- Une ligne par patiente -------------------------------------------
    # Si une patiente porte plusieurs labels (lésions multiples), on retient le
    # plus sévère : cliniquement, une patiente avec une lésion maligne est un cas.
    patients = (
        df.groupby(group_col)[label_col].max().reset_index().rename(columns={label_col: "_y"})
    )
    y = patients["_y"].to_numpy()
    n = len(patients)

    # --- Antécédents familiaux --------------------------------------------
    p_fh_case = _prob_from_or(priors.family_history_prev_control, priors.family_history_or)
    p_fh = np.where(y == 1, p_fh_case, priors.family_history_prev_control)
    patients["family_history"] = (rng.random(n) < p_fh).astype(int)

    # --- BRCA1/2 -----------------------------------------------------------
    p_brca = np.where(y == 1, priors.brca_prev_case, priors.brca_prev_control)
    patients["brca"] = (rng.random(n) < p_brca).astype(int)

    # --- IMC ---------------------------------------------------------------
    bmi = rng.normal(
        priors.bmi_mean_control + priors.bmi_shift_case * y,
        priors.bmi_sd,
        size=n,
    )
    patients["bmi"] = np.clip(bmi, 15.0, 50.0).round(1)

    # --- Densité BI-RADS ---------------------------------------------------
    p_ctrl = np.asarray(priors.density_probs_control, dtype=float)
    p_ctrl = p_ctrl / p_ctrl.sum()
    # On réhausse les catégories denses (C, D) chez les cas selon l'OR.
    w = np.array([1.0, 1.0, priors.density_or_dense, priors.density_or_dense])
    p_case = p_ctrl * w
    p_case = p_case / p_case.sum()

    density = np.empty(n, dtype=object)
    is_case = y == 1
    if is_case.any():
        density[is_case] = rng.choice(DENSITY_LEVELS, size=int(is_case.sum()), p=p_case)
    if (~is_case).any():
        density[~is_case] = rng.choice(DENSITY_LEVELS, size=int((~is_case).sum()), p=p_ctrl)
    patients["density_synth"] = density

    # --- Âge (optionnel) ---------------------------------------------------
    if fill_missing_age:
        age = rng.normal(
            priors.age_mean_control + priors.age_shift_case * y, priors.age_sd, size=n
        )
        patients["age_synth"] = np.clip(age, priors.age_min, priors.age_max).round(0)

    patients = patients.drop(columns=["_y"])

    # --- Report sur le manifeste ------------------------------------------
    out = df.merge(patients, on=group_col, how="left", validate="many_to_one")

    if fill_missing_age and "age" in out.columns:
        n_before = int(out["age"].isna().sum())
        out["age"] = out["age"].fillna(out["age_synth"])
        out["age_is_synthetic"] = out["age"].notna() & df["age"].isna().to_numpy()
        print(f"  Âges complétés par génération : {n_before}")

    out["clinical_level"] = "C2"
    if "age" in out.columns and out["age"].notna().any():
        out.loc[out["age"].notna(), "clinical_level"] = "C0+C2"

    return out


# ---------------------------------------------------------------------------
# Audit anti-fuite
# ---------------------------------------------------------------------------
def audit_leakage(
    df: pd.DataFrame,
    clinical_cols: list[str] | None = None,
    label_col: str = "label",
    group_col: str = "patient_id",
    max_auc: float = DEFAULT_MAX_AUC,
    strict: bool = True,
    seed: int = 42,
) -> dict:
    """Mesure le pouvoir discriminant des seules variables cliniques.

    Entraîne une régression logistique sur les variables cliniques uniquement,
    en validation croisée **groupée par patiente**, et compare l'AUC obtenue au
    plafond. Une régression logistique est volontairement choisie : elle est
    faible, donc si *elle* dépasse le plafond, la fuite est massive.

    Un rapport par variable est également produit — il permet d'identifier
    immédiatement laquelle fuit.

    Raises
    ------
    ClinicalLeakageError
        Si ``strict`` et que l'AUC dépasse ``max_auc``.
    """
    clinical_cols = clinical_cols or ["family_history", "brca", "bmi", "density_synth"]
    cols = [c for c in clinical_cols if c in df.columns]
    if not cols:
        raise KeyError(f"Aucune des colonnes cliniques {clinical_cols} n'est présente.")

    # Une ligne par patiente : sinon les patientes à nombreuses images pèsent
    # davantage et l'AUC mesurée n'a pas le sens attendu.
    agg = {c: "first" for c in cols}
    agg[label_col] = "max"
    per_patient = df.groupby(group_col).agg(agg).reset_index()

    X = pd.get_dummies(per_patient[cols], drop_first=True).astype(float)
    y = per_patient[label_col].to_numpy()
    groups = per_patient[group_col].to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("Un seul label présent — audit impossible.")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("Trop peu d'exemples dans une classe pour auditer.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))

    # Contribution individuelle de chaque variable — identifie la coupable.
    per_var: dict[str, float] = {}
    for c in cols:
        Xc = pd.get_dummies(per_patient[[c]], drop_first=True).astype(float)
        if Xc.shape[1] == 0:
            continue
        pc = cross_val_predict(
            model, Xc, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
        per_var[c] = round(float(roc_auc_score(y, pc)), 4)

    passed = auc <= max_auc
    report = {
        "auc_clinical_only": round(auc, 4),
        "max_auc": max_auc,
        "passed": passed,
        "n_patients": int(len(per_patient)),
        "n_positive": int(y.sum()),
        "per_variable_auc": per_var,
        "columns": cols,
    }

    status = "✅ OK" if passed else "❌ FUITE"
    print(f"\n{'─' * 62}\nAUDIT ANTI-FUITE CLINIQUE\n{'─' * 62}")
    print(f"  Patientes                : {report['n_patients']} ({report['n_positive']} malignes)")
    print(f"  AUC — variables cliniques seules : {auc:.4f}   (plafond {max_auc})   {status}")
    print("  Contribution par variable :")
    for c, a in sorted(per_var.items(), key=lambda kv: -kv[1]):
        flag = "  ← principale contributrice" if a == max(per_var.values()) else ""
        print(f"      {c:<18} AUC {a:.4f}{flag}")

    if not passed:
        print(
            "\n  ⚠️  Le générateur encode le label trop fortement.\n"
            "     Réduisez les tailles d'effet dans ClinicalPriors (odds ratios,\n"
            "     décalages de moyenne) et régénérez. N'utilisez EN AUCUN CAS\n"
            "     ces données pour un résultat publié."
        )
        if strict:
            raise ClinicalLeakageError(
                f"AUC clinique seule = {auc:.4f} > plafond {max_auc}. "
                "Voir src/data/clinical_synth.py, RÈGLE N°2."
            )
    print(f"{'─' * 62}\n")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère les variables cliniques C2 et audite la fuite de cible."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-auc", type=float, default=DEFAULT_MAX_AUC)
    parser.add_argument("--fill-missing-age", action="store_true")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="N'échoue pas en cas de fuite (diagnostic uniquement).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Manifeste : {len(df)} images · {df['patient_id'].nunique()} patientes")
    print(f"Paramètres épidémiologiques : {asdict(ClinicalPriors())}")

    out = generate(df, seed=args.seed, fill_missing_age=args.fill_missing_age)
    audit_leakage(out, max_auc=args.max_auc, strict=not args.no_strict, seed=args.seed)

    out_path = args.out or args.manifest
    out.to_csv(out_path, index=False)
    print(f"✅ Écrit dans {out_path}")


if __name__ == "__main__":
    main()
