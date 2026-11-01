"""
Construction du **manifeste unifié** — le fichier pivot du projet.

Six datasets, six conventions de nommage, six façons d'encoder le label. Ce
module les réduit à un seul CSV. Tout le reste du pipeline (DataLoaders,
splits, entraînement, évaluation) lit `data/manifest.csv` et **rien d'autre** :
aucun module en aval ne doit jamais parcourir les dossiers bruts.

Bénéfice concret : ajouter INbreast à sa réception ne demande qu'un parser de
30 lignes, sans toucher au code de modélisation.

Schéma de sortie
----------------
    patient_id      str    PRÉFIXÉ PAR LE DATASET (cbis_P00123)
    image_path      str    relatif à data/raw/, séparateur `/` portable
    dataset         str    cbis_ddsm | mias | busi | cmmd | inbreast | breast
    modality        str    mammography | ultrasound
    view            str    CC | MLO | US | unknown
    laterality      str    L | R | unknown
    label           int    0 = bénin, 1 = malin
    lesion_type     str    mass | calcification | both | unknown
    age             float  RÉEL si disponible, sinon NaN
    density         str    BI-RADS A/B/C/D réel, sinon vide
    mask_path       str    masque de lésion, sinon vide
    clinical_level  str    C0 | C1 | C2

⚠️ Le préfixe de `patient_id` n'est pas cosmétique. CBIS-DDSM et CMMD utilisent
tous deux des identifiants numériques courts : sans préfixe, la patiente « 42 »
de CBIS et la patiente « 42 » de CMMD fusionnent, et le découpage par patiente
devient silencieusement faux.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCHEMA = [
    "patient_id", "image_path", "dataset", "modality", "view", "laterality",
    "label", "lesion_type", "age", "density", "mask_path", "clinical_level",
]

_PARSERS: dict[str, callable] = {}


def register(name: str):
    """Enregistre un parser de dataset. Ajouter un dataset = ajouter une fonction."""
    def deco(fn):
        _PARSERS[name] = fn
        return fn
    return deco


def _empty_row(**kw) -> dict:
    row = {
        "patient_id": "", "image_path": "", "dataset": "", "modality": "",
        "view": "unknown", "laterality": "unknown", "label": -1,
        "lesion_type": "unknown", "age": np.nan, "density": "",
        "mask_path": "", "clinical_level": "C0",
    }
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# CBIS-DDSM
# ---------------------------------------------------------------------------
@register("cbis_ddsm")
def parse_cbis_ddsm(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Parse la version Kaggle (awsaf49) de CBIS-DDSM.

    LE PIÈGE DOCUMENTÉ, CONFIRMÉ SUR NOS FICHIERS
    ---------------------------------------------
    Les CSV de cas référencent des chemins DICOM (`.../000000.dcm`) qui
    n'existent pas dans ce téléchargement : les images réelles sont des JPEG
    sous `jpeg/<SeriesInstanceUID>/*.jpg`. La correspondance passe par
    `dicom_info.csv` : on extrait le SeriesInstanceUID (2e répertoire du chemin
    DICOM annoncé) et on le résout via ce fichier.

    `SeriesDescription` y distingue trois types de séries :
        'full mammogram images'  → l'image d'entrée du modèle
        'ROI mask images'        → le masque (validation IoU du Grad-CAM)
        'cropped images'         → ignoré (ROI recadrée, redondante)

    Sur le label : `pathology` ∈ {MALIGNANT, BENIGN, BENIGN_WITHOUT_CALLBACK}.
    BENIGN_WITHOUT_CALLBACK est classé bénin (0) — choix à documenter dans le
    rapport, certains travaux excluent ces cas.
    """
    csv_dir = raw_dir / "csv"
    info_path = csv_dir / "dicom_info.csv"
    if not info_path.exists():
        warnings.warn(f"[cbis_ddsm] dicom_info.csv absent : {info_path}")
        return pd.DataFrame()

    info = pd.read_csv(info_path, low_memory=False)
    info["jpeg_rel"] = (
        info["image_path"].astype(str).str.replace("CBIS-DDSM/", "", regex=False)
    )
    # SeriesInstanceUID → (chemin jpeg, description de série)
    series_map: dict[str, tuple[str, str]] = {}
    for _, r in info.iterrows():
        series_map[str(r["SeriesInstanceUID"])] = (
            str(r["jpeg_rel"]), str(r.get("SeriesDescription", "")).lower()
        )

    def resolve(dicom_path: str, want: str) -> str:
        """Chemin annoncé dans le CSV de cas → jpeg réel du type voulu."""
        parts = str(dicom_path).strip().split("/")
        # format : <descriptif>/<StudyUID>/<SeriesUID>/000000.dcm
        for p in parts:
            hit = series_map.get(p)
            if hit and want in hit[1]:
                return hit[0]
        return ""

    rows = []
    for csv_name in cfg.get("csv_files", []):
        path = csv_dir / csv_name
        if not path.exists():
            warnings.warn(f"[cbis_ddsm] absent : {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        is_calc = "calc" in csv_name

        for _, r in df.iterrows():
            img = resolve(r.get("image_file_path", ""), "full")
            if not img:
                continue        # série absente du téléchargement
            mask = resolve(r.get("roi_mask_file_path", ""), "mask")

            rows.append(_empty_row(
                patient_id=f"cbis_{r['patient_id']}",
                image_path=img,
                dataset="cbis_ddsm",
                modality="mammography",
                view=str(r.get("image_view", "unknown")).upper(),
                laterality=str(r.get("left_or_right_breast", "u"))[0].upper(),
                label=1 if str(r.get("pathology", "")).upper() == "MALIGNANT" else 0,
                lesion_type="calcification" if is_calc else "mass",
                density=_cbis_density(r.get("breast_density")),
                mask_path=mask,
                clinical_level="C0",
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Plusieurs anomalies peuvent partager la même image pleine (abnormality
    # id 1, 2...) : une ligne par image, label le plus sévère.
    n = len(df)
    df = (df.sort_values("label", ascending=False)
            .drop_duplicates(subset="image_path", keep="first")
            .reset_index(drop=True))
    if len(df) < n:
        print(f"  [cbis_ddsm] {n - len(df)} anomalies multiples agrégées au niveau image")
    return df


def _cbis_density(v) -> str:
    try:
        return {1: "A", 2: "B", 3: "C", 4: "D"}.get(int(v), "")
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# MIAS
# ---------------------------------------------------------------------------
@register("mias")
def parse_mias(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Parse `Info.txt` de la base MIAS.

    Format : `mdb001 G CIRC B 535 425 197`
        refnum · tissu (F/G/D) · classe d'anomalie · sévérité (B/M) · x · y · rayon

    ⚠️ Deux subtilités qui coûtent cher si on les rate :

    1. **Les images vont par paires.** mdb001/mdb002 sont les seins gauche et
       droit de la MÊME patiente. Le `patient_id` doit donc être `(refnum-1)//2`
       et non le numéro d'image, sinon fuite garantie entre train et test.
    2. Les cas NORM n'ont que 4 champs. Ils sont exclus (tâche binaire).
    """
    info = raw_dir / cfg.get("info_file", "Info.txt")
    if not info.exists():
        warnings.warn(f"[mias] absent : {info}")
        return pd.DataFrame()

    rows = []
    for line in info.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 4 or not re.match(r"^mdb\d+", parts[0]):
            continue
        refnum, _tissue, abnorm, severity = parts[0], parts[1], parts[2], parts[3]
        if abnorm.upper() == "NORM" or severity.upper() not in ("B", "M"):
            continue

        n = int(refnum[3:])
        patient = (n - 1) // 2          # ← paires gauche/droite
        rows.append(_empty_row(
            patient_id=f"mias_P{patient:04d}",
            image_path=f"{refnum}.pgm",
            dataset="mias",
            modality="mammography",
            view="MLO",                 # MIAS est intégralement en vue MLO
            laterality="R" if n % 2 == 1 else "L",
            label=1 if severity.upper() == "M" else 0,
            lesion_type="calcification" if abnorm.upper() == "CALC" else "mass",
            clinical_level="C0",
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Certaines images portent PLUSIEURS lésions (Info.txt contient ~330
    # entrées pour 322 images) : une ligne par lésion créerait des doublons
    # d'image. On agrège au niveau image — label le plus sévère, comme pour
    # les patientes multi-lésions de CMMD.
    n_before = len(df)
    df = (
        df.sort_values("label", ascending=False)
        .drop_duplicates(subset="image_path", keep="first")
        .reset_index(drop=True)
    )
    if len(df) < n_before:
        print(f"  [mias] {n_before - len(df)} lésion(s) multiple(s) agrégée(s) "
              f"au niveau image (label le plus sévère retenu)")
    return df


# ---------------------------------------------------------------------------
# BUSI
# ---------------------------------------------------------------------------
@register("busi")
def parse_busi(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Parse BUSI depuis l'arborescence `benign/ malignant/ normal/`.

    ⚠️ LIMITATION IMPORTANTE À DOCUMENTER DANS LE RAPPORT
    ------------------------------------------------------
    BUSI compte 780 images provenant de 600 patientes, mais **la correspondance
    image → patiente n'est pas publiée**. Il est donc impossible de garantir
    l'absence totale de fuite au sein de BUSI : deux images de la même femme
    peuvent se retrouver de part et d'autre du split.

    On traite chaque image comme une patiente distincte, ce qui est la seule
    option disponible — mais c'est une hypothèse, pas une garantie. Environ
    180 images (23 %) sont potentiellement des doublons de patiente.

    Ne pas passer cette limite sous silence : elle explique une partie des AUC
    très élevées (> 0,99) rapportées sur BUSI dans la littérature, y compris
    chez Latha et al. (2024) que cite votre état de l'art.
    """
    exclude = {c.lower() for c in cfg.get("exclude_classes", ["normal"])}
    rows = []
    for cls_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name.lower()
        if cls in exclude or cls not in ("benign", "malignant"):
            continue
        for img in sorted(cls_dir.glob("*.png")):
            if "_mask" in img.stem:
                continue
            mask = img.with_name(f"{img.stem}_mask.png")
            rows.append(_empty_row(
                patient_id=f"busi_{cls}_{img.stem.replace(' ', '_')}",
                image_path=str(img.relative_to(raw_dir)),
                dataset="busi",
                modality="ultrasound",
                view="US",
                label=1 if cls == "malignant" else 0,
                mask_path=str(mask.relative_to(raw_dir)) if mask.exists() else "",
                clinical_level="C0",
            ))

    if rows:
        warnings.warn(
            "[busi] Correspondance image→patiente non publiée : chaque image est "
            "traitée comme une patiente distincte. Fuite intra-BUSI possible "
            "(~23 % des images). À signaler dans les limites du rapport."
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BrEaST (Pologne) — remplaçant d'UDIAT, avec âge et BI-RADS RÉELS
# ---------------------------------------------------------------------------
@register("breast_poland")
def parse_breast_poland(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Parse BrEaST-Lesions-USG — structure réelle vérifiée le 8 août 2026.

        breast/
        ├── BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
        └── BrEaST-Lesions_USG-images_and_masks/
            ├── case001.png
            ├── case001_tumor.png        ← masque
            └── ...

    Colonnes réelles du xlsx : CaseID, Image_filename, Mask_tumor_filename,
    Age, Classification (benign/malignant/normal), BIRADS, Tissue_composition…

    Particularités observées :
    - `Age` contient parfois la chaîne "not available" → NaN via _to_float.
    - `BIRADS` est en notation 4a/4b/4c — conservé tel quel dans `density`
      (l'encodeur one-hot le gère comme catégoriel).
    - 1 cas = 1 patiente : l'âge est RÉEL, c'est le pendant écho de CMMD.
    """
    candidates = list(raw_dir.glob("*clinical*.xls*"))
    if not candidates:
        warnings.warn(f"[breast_poland] fichier clinique introuvable sous {raw_dir}")
        return pd.DataFrame()
    clin = candidates[0]

    img_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and "images" in d.name.lower()]
    img_dir = img_dirs[0] if img_dirs else raw_dir

    df = pd.read_excel(clin)
    df.columns = [c.strip() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        cls = str(r.get("Classification", "")).strip().lower()
        if cls not in ("benign", "malignant"):
            continue
        fname = str(r.get("Image_filename", "")).strip()
        if not fname or not (img_dir / fname).exists():
            continue
        mask = str(r.get("Mask_tumor_filename", "")).strip()
        mask_rel = f"{img_dir.name}/{mask}" if mask and (img_dir / mask).exists() else ""

        rows.append(_empty_row(
            patient_id=f"brpl_case{str(r.get('CaseID', Path(fname).stem)).strip()}",
            image_path=f"{img_dir.name}/{fname}",
            dataset="breast_poland",
            modality="ultrasound",
            view="US",
            label=1 if cls == "malignant" else 0,
            age=_to_float(r.get("Age")),
            density=str(r.get("BIRADS", "")).strip(),
            mask_path=mask_rel,
            clinical_level="C0",          # ← âge et BI-RADS RÉELS
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BUS-UCLM
# ---------------------------------------------------------------------------
@register("bus_uclm")
def parse_bus_uclm(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Parse BUS-UCLM — structure réelle vérifiée le 8 août 2026.

        bus-uclm/BUS-UCLM Breast ultrasound lesion segmentation dataset/
        └── BUS-UCLM/
            ├── INFO.csv          (séparateur ';', BOM UTF-8)
            ├── images/ALWI_000.png ...
            └── masks/  (RGB : vert=bénin, rouge=malin, noir=fond)

    INFO.csv : colonnes Image;Resolution;Label;Doppler;Marks;Combined
    avec Label ∈ {Normal(419), Benign(174), Malignant(90)} — vérifié.

    **La patiente est le préfixe du nom de fichier** (ALWI_000 → ALWI) :
    38 patientes pour 683 images, soit ~18 images par patiente. C'est LE
    dataset où le regroupement importe le plus — sans lui, les images d'une
    même femme se répartiraient entre train et test.

    Rôle dans le projet : jeu de TEST EXTERNE des configs de généralisation
    échographie (264 images utiles, trop petit pour entraîner).
    """
    inner = next(raw_dir.rglob("INFO.csv"), None)
    if inner is None:
        warnings.warn(f"[bus_uclm] INFO.csv introuvable sous {raw_dir}")
        return pd.DataFrame()
    base = inner.parent

    df = pd.read_csv(inner, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        cls = str(r.get("Label", "")).strip().lower()
        if cls not in ("benign", "malignant"):
            continue                                  # exclut Normal (419)
        fname = str(r.get("Image", "")).strip()
        img = base / "images" / fname
        if not img.exists():
            continue
        mask = base / "masks" / fname
        rel = base.relative_to(raw_dir)

        rows.append(_empty_row(
            patient_id=f"uclm_{fname.split('_')[0]}",   # ← préfixe = patiente
            image_path=str(rel / "images" / fname),
            dataset="bus_uclm",
            modality="ultrasound",
            view="US",
            label=1 if cls == "malignant" else 0,
            mask_path=str(rel / "masks" / fname) if mask.exists() else "",
            clinical_level="C0",
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        n_pat = out["patient_id"].nunique()
        print(f"  [bus_uclm] {len(out)} images / {n_pat} patientes "
              f"(~{len(out) / n_pat:.0f} images/patiente — regroupement crucial)")
        # ⚠️ masques RGB : la conversion en binaire se fait au chargement
        # (BreastDataset lit en niveaux de gris > 127, ce qui capte vert et rouge).
    return out


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retourne le premier nom de colonne présent parmi les candidats."""
    return next((c for c in candidates if c in df.columns), None)


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# CMMD — source principale de données cliniques RÉELLES
# ---------------------------------------------------------------------------
@register("cmmd")
def parse_cmmd(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Parse CMMD — structure réelle vérifiée le 8 août 2026.

        cmmd/
        ├── CMMD_clinicaldata_revision.xlsx   (ID1, LeftRight, Age, number,
        │                                      abnormality, classification, subtype)
        └── CMMD/cmmd/D1-0001/<uid>/<uid>/*.dcm   (2 à 4 DICOM par patiente)

    Le xlsx donne le diagnostic PAR SEIN (ID1 + LeftRight) : une patiente peut
    avoir un sein bénin et l'autre malin. Il faut donc lire l'en-tête DICOM de
    chaque fichier (tags Laterality et ViewPosition, sans charger les pixels —
    rapide) pour associer chaque image au bon diagnostic.

    C'est le dataset le plus précieux du projet : l'ÂGE RÉEL de chaque patiente
    rend la fusion clinique C0 possible.
    """
    import pydicom

    clin = next(raw_dir.glob("*clinicaldata*.xls*"), None)
    if clin is None:
        warnings.warn(f"[cmmd] fichier clinique absent sous {raw_dir}")
        return pd.DataFrame()

    df = pd.read_excel(clin)
    df.columns = [c.strip() for c in df.columns]

    # (ID1, côté) → ligne clinique
    clinical: dict[tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        cls = str(r.get("classification", "")).strip().lower()
        if cls not in ("benign", "malignant"):
            continue
        pid = str(r["ID1"]).strip()
        side = str(r.get("LeftRight", "")).strip().upper()[:1]
        clinical[(pid, side)] = {
            "label": 1 if cls == "malignant" else 0,
            "age": _to_float(r.get("Age")),
            "lesion": {"calcification": "calcification", "mass": "mass",
                       "both": "both"}.get(
                str(r.get("abnormality", "")).strip().lower(), "unknown"),
        }

    # Localise la racine des dossiers patientes (imbrication CMMD/cmmd/ observée)
    patient_root = None
    for cand in [raw_dir / "CMMD" / "cmmd", raw_dir / "CMMD", raw_dir]:
        if cand.exists() and any(d.name.startswith("D") for d in cand.iterdir() if d.is_dir()):
            patient_root = cand
            break
    if patient_root is None:
        warnings.warn(f"[cmmd] dossiers patientes (D1-XXXX) introuvables sous {raw_dir}")
        return pd.DataFrame()

    rows, skipped = [], 0
    for pdir in sorted(patient_root.iterdir()):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        for dcm in sorted(pdir.rglob("*.dcm")):
            try:
                # stop_before_pixels : on ne lit que l'en-tête (~1 ms/fichier)
                ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
                side = str(getattr(ds, "ImageLaterality",
                                   getattr(ds, "Laterality", ""))).strip().upper()[:1]
                view = str(getattr(ds, "ViewPosition", "unknown")).strip().upper()
            except Exception:
                skipped += 1
                continue

            info = clinical.get((pid, side))
            if info is None:
                skipped += 1        # sein sans diagnostic dans le xlsx
                continue

            rows.append(_empty_row(
                patient_id=f"cmmd_{pid}",
                image_path=str(dcm.relative_to(raw_dir)),
                dataset="cmmd",
                modality="mammography",
                view=view if view in ("CC", "MLO") else "unknown",
                laterality=side or "unknown",
                label=info["label"],
                lesion_type=info["lesion"],
                age=info["age"],
                clinical_level="C0",     # ← âge RÉEL
            ))

    if skipped:
        print(f"  [cmmd] {skipped} DICOM ignorés (côté sans diagnostic ou en-tête illisible)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Assemblage et validation
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Contrôles d'intégrité — échouer ici coûte bien moins cher qu'à l'entraînement."""
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    if not df["label"].isin([0, 1]).all():
        bad = df.loc[~df["label"].isin([0, 1]), "label"].unique()
        raise ValueError(f"Labels invalides : {bad}")

    if df["patient_id"].eq("").any():
        raise ValueError("Des patient_id sont vides.")

    # Le préfixe dataset est ce qui empêche deux datasets de partager un identifiant.
    unprefixed = df.loc[~df["patient_id"].str.contains("_", regex=False), "patient_id"]
    if len(unprefixed):
        raise ValueError(f"patient_id sans préfixe dataset : {unprefixed.unique()[:5]}")

    # Une patiente ne doit pas appartenir à deux datasets.
    per_pid = df.groupby("patient_id")["dataset"].nunique()
    if (per_pid > 1).any():
        raise ValueError(
            f"Collision d'identifiants entre datasets : {per_pid[per_pid > 1].index[:5].tolist()}"
        )

    dup = df.duplicated(subset=["dataset", "image_path"]).sum()
    if dup:
        warnings.warn(f"{dup} ligne(s) image dupliquée(s) — vérifiez les parsers.")


def build(config_path: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frames = []

    for name, dcfg in cfg["datasets"].items():
        if not dcfg.get("enabled", False):
            print(f"  ⏭  {name:<12} désactivé")
            continue
        parser = _PARSERS.get(name)
        if parser is None:
            warnings.warn(f"Aucun parser pour '{name}' — ignoré.")
            continue

        raw_dir = Path(dcfg["raw_dir"])
        if not raw_dir.exists():
            print(f"  ⚠  {name:<12} dossier absent ({raw_dir}) — ignoré")
            continue

        df = parser(dcfg, raw_dir)
        if df.empty:
            print(f"  ⚠  {name:<12} 0 ligne")
            continue

        # Préfixe = nom réel du dossier (peut différer de la clé de config :
        # CBIS-DDSM, bus-uclm...). Les chemins du manifeste sont ainsi tous
        # relatifs à data/raw/, la racine unique des DataLoaders.
        prefix = raw_dir.name
        df["image_path"] = (prefix + "/" + df["image_path"].astype(str)).str.replace(
            "\\", "/", regex=False
        )
        # Le masque suit la même convention — sans quoi la validation IoU du
        # Grad-CAM (Phase 4) chercherait au mauvais endroit.
        mask_paths = df["mask_path"].fillna("").astype(str)
        nonempty = mask_paths.str.len() > 0
        df["mask_path"] = ""
        df.loc[nonempty, "mask_path"] = (
            prefix + "/" + mask_paths[nonempty]
        ).str.replace("\\", "/", regex=False)
        frames.append(df[SCHEMA])
        print(
            f"  ✅ {name:<12} {len(df):>6} images · "
            f"{df['patient_id'].nunique():>5} patientes · "
            f"{df['label'].mean():.1%} malignes"
        )

    if not frames:
        raise RuntimeError(
            "Aucun dataset chargé. Vérifiez `enabled` et les chemins dans configs/data.yaml."
        )

    manifest = pd.concat(frames, ignore_index=True)
    validate(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit le manifeste unifié.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()

    print("Construction du manifeste\n" + "─" * 62)
    manifest = build(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    print("─" * 62)
    print(f"TOTAL   {len(manifest)} images · {manifest['patient_id'].nunique()} patientes")
    print(f"        {manifest['label'].sum()} malignes ({manifest['label'].mean():.1%})")
    print("\nPar modalité :")
    print(manifest.groupby("modality").agg(
        images=("label", "size"),
        patientes=("patient_id", "nunique"),
        taux_malin=("label", "mean"),
    ).round(3).to_string())
    print(f"\nÂges réels disponibles : {manifest['age'].notna().sum()} lignes")
    print(f"✅ Écrit dans {args.out}")


if __name__ == "__main__":
    main()
