"""
Datasets PyTorch — mono-modal et multi-modal.

Trois points de conception méritent l'attention.

**1. Pas de flip vertical.** L'augmentation par défaut de beaucoup de pipelines
inclut `RandomVerticalFlip`. C'est une erreur en mammographie : les vues CC et
MLO ont une orientation anatomique définie (le muscle pectoral apparaît en haut
en MLO, le mamelon est orienté). Retourner verticalement produit des images
anatomiquement impossibles. Le flip horizontal, lui, est légitime : il revient
à passer d'un sein gauche à un sein droit.

**2. Normalisation des variables cliniques ajustée sur le TRAIN uniquement.**
Calculer moyenne et écart-type sur l'ensemble du dataset est une fuite discrète
mais réelle : les statistiques du jeu de test influencent l'entraînement.
`ClinicalEncoder.fit()` ne doit voir que le split d'entraînement.

**3. L'appariement virtuel est explicite et rejouable.** Par défaut,
`PairedDataset` associe aléatoirement une mammographie et une échographie sans
consulter leurs labels — ce sont des patientes DIFFÉRENTES. L'appariement
conditionné par label existe uniquement comme option explicite pour les
analyses descriptives, jamais pour la validation ou le test. Le seed est
toujours un paramètre afin que les tirages soient reproductibles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:                                   # permet d'importer sans torch
    torch = None
    Dataset = object


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
def build_transforms(cfg: dict, train: bool):
    """Construit le pipeline albumentations.

    ⚠️ `VerticalFlip` est délibérément absent — voir le point 1 du docstring
    du module. Si vous l'ajoutez, documentez pourquoi.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    size = cfg["preprocessing"]["image_size"]
    norm = cfg["preprocessing"]["normalize"]

    if not train:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=norm["mean"], std=norm["std"]),
            ToTensorV2(),
        ])

    aug = cfg["augmentation"]
    if aug.get("vflip_prob", 0.0) > 0:
        raise ValueError(
            "vflip_prob > 0 : le flip vertical produit des mammographies "
            "anatomiquement impossibles (orientation CC/MLO). "
            "Si le choix est délibéré, retirez ce garde-fou explicitement."
        )

    # ⚠️ per_channel=False est OBLIGATOIRE ici.
    # Les images médicales sont en niveaux de gris, dupliquées sur 3 canaux.
    # Un bruit tiré indépendamment par canal produit du bruit COLORÉ : des
    # pixels rouges/verts/bleus qui n'existent dans aucune échographie réelle.
    # Le modèle apprend alors à ignorer un artefact absent du test, et une
    # partie de la capacité du réseau part dans le vide.
    # Détecté le 8 août 2026 par le contrôle visuel de la cellule 5 du notebook
    # — aucun test automatique ne l'aurait vu.
    try:
        noise = A.GaussNoise(per_channel=False, p=aug.get("gauss_noise_prob", 0.2))
    except TypeError:                      # albumentations très ancien
        noise = A.GaussNoise(p=aug.get("gauss_noise_prob", 0.2))

    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=aug["hflip_prob"]),
        A.Rotate(limit=aug["rotation_deg"], border_mode=0, p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_contrast"],
            contrast_limit=aug["brightness_contrast"],
            p=0.5,
        ),
        noise,
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Encodage des variables cliniques
# ---------------------------------------------------------------------------
@dataclass
class ClinicalEncoder:
    """Normalise les continues (z-score) et encode les catégorielles (one-hot).

    ⚠️ `fit()` NE DOIT VOIR QUE LE SPLIT D'ENTRAÎNEMENT. Ajuster les statistiques
    sur l'ensemble des données fait fuiter de l'information du test vers le
    train. L'effet est faible mais réel, et c'est exactement le genre de détail
    qu'un relecteur attentif relève.
    """

    continuous: list[str] = field(default_factory=lambda: ["age", "bmi"])
    binary: list[str] = field(default_factory=lambda: ["family_history", "brca"])
    categorical: list[str] = field(default_factory=lambda: ["density"])

    means_: dict = field(default_factory=dict, init=False)
    stds_: dict = field(default_factory=dict, init=False)
    categories_: dict = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame) -> "ClinicalEncoder":
        for c in self.continuous:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                self.means_[c] = float(vals.mean()) if len(vals) else 0.0
                self.stds_[c] = float(vals.std()) or 1.0
        for c in self.categorical:
            if c in df.columns:
                cats = sorted({str(v) for v in df[c].dropna() if str(v).strip()})
                self.categories_[c] = cats
        self.fitted_ = True
        return self

    @property
    def n_features(self) -> int:
        n = len(self.means_) * 2          # valeur + indicateur de présence
        n += len(self.binary)
        n += sum(len(v) for v in self.categories_.values())
        return n

    def transform_row(self, row: pd.Series) -> np.ndarray:
        """Encode une ligne. Les manquants → 0 + indicateur de présence à 0.

        L'indicateur de présence est important : il permet au modèle de
        distinguer « âge inconnu » de « âge moyen ». Sans lui, imputer par la
        moyenne revient à affirmer une information qu'on n'a pas.
        """
        if not self.fitted_:
            raise RuntimeError("ClinicalEncoder.fit() n'a pas été appelé.")

        feats: list[float] = []
        for c in self.means_:
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.isna(v):
                feats += [0.0, 0.0]
            else:
                feats += [(float(v) - self.means_[c]) / self.stds_[c], 1.0]

        for c in self.binary:
            v = row.get(c)
            feats.append(0.0 if pd.isna(v) else float(v))

        for c, cats in self.categories_.items():
            v = str(row.get(c, ""))
            feats += [1.0 if v == cat else 0.0 for cat in cats]

        return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset mono-modal
# ---------------------------------------------------------------------------
class BreastDataset(Dataset):
    """Une modalité (mammographie OU échographie) + variables cliniques.

    Retourne un dictionnaire :
        image     FloatTensor (3, H, W)
        clinical  FloatTensor (n_features,)
        label     FloatTensor scalaire
        mask      FloatTensor (H, W)  — présent seulement si return_mask
        index     int                 — pour retrouver la ligne du manifeste
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        root: Path,
        transform=None,
        clinical_encoder: ClinicalEncoder | None = None,
        return_mask: bool = False,
    ):
        self.df = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.enc = clinical_encoder
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> np.ndarray:
        """Charge une image et la ramène en RGB uint8.

        Les PNG 16 bits produits par dicom_convert sont ramenés en 8 bits ici
        seulement — le stockage reste en 16 bits pour ne pas perdre
        d'information de façon irréversible.
        """
        import cv2

        path = self.root / rel_path
        if path.suffix.lower() == ".dcm":
            # CMMD est stocké en DICOM : lecture directe avec LUT (voir
            # dicom_convert). Pour l'entraînement intensif, préférer la
            # conversion préalable en PNG — la lecture DICOM est ~5× plus lente.
            from src.data.dicom_convert import dicom_to_array
            img = dicom_to_array(path)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image illisible : {path}")

        if img.dtype == np.uint16:
            img = (img / 257.0).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int) -> dict:
        import cv2

        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])

        mask = None
        if self.return_mask and str(row.get("mask_path", "")).strip():
            # Lecture en couleur puis max sur les canaux : les masques BUSI sont
            # binaires (blanc), ceux de BUS-UCLM sont RGB (vert=bénin,
            # rouge=malin). Un simple IMREAD_GRAYSCALE perdrait le rouge :
            # 0,299×255 ≈ 76 < 127 → le masque des malignes disparaîtrait.
            m = cv2.imread(str(self.root / row["mask_path"]), cv2.IMREAD_COLOR)
            if m is not None:
                mask = (m.max(axis=2) > 127).astype(np.uint8)

        if self.transform is not None:
            out = (
                self.transform(image=img, mask=mask)
                if mask is not None
                else self.transform(image=img)
            )
            img = out["image"]
            mask = out.get("mask")

        sample = {
            "image": img,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "index": idx,
        }
        if self.enc is not None:
            sample["clinical"] = torch.from_numpy(self.enc.transform_row(row))
        if self.return_mask:
            sample["mask"] = (
                mask.float() if mask is not None
                else torch.zeros(img.shape[-2:], dtype=torch.float32)
            )
        return sample


# ---------------------------------------------------------------------------
# Dataset apparié — PISTE B
# ---------------------------------------------------------------------------
class PairedDataset(Dataset):
    """Appariement **virtuel** mammographie ↔ échographie.

    ⚠️ LIRE AVANT UTILISATION
    -------------------------
    Les deux images d'une paire proviennent de **patientes différentes**. Aucun
    dataset public n'apparie les deux modalités chez la même femme (voir
    00_PLAN_PROJET.md, Verrou n°1). Par défaut, l'association est aléatoire et
    indépendante des labels de chaque split.

    Conséquence directe : le modèle ne peut pas apprendre une véritable
    interaction inter-modale, puisqu'il n'y en a aucune dans les données. C'est
    pourquoi le protocole impose deux comparaisons obligatoires :

      - contre la **moyenne des logits** des deux modèles mono-modaux ;
      - contre un appariement **aléatoire ignorant le label** (test d'intégrité).

    L'appariement conditionné par label est disponible uniquement à la demande
    pour construire un jeu d'entraînement virtuel ; il ne doit jamais servir
    à créer les paires de validation ou de test, car cela révèle leur label.

    Si la cross-attention ne bat pas la moyenne des logits, elle n'apporte rien.
    C'est un résultat publiable et honnête, pas un échec.

    Le protocole demande 5 tirages (`pair_seed` de 0 à 4) avec écart-type
    reporté : une forte variance signalerait que le modèle apprend l'appariement
    plutôt que la pathologie. Les paires restent dans l'ordre mammographique
    afin d'aligner les prédictions de fusion sur les baselines ; le DataLoader
    mélange les exemples d'entraînement séparément.
    """

    def __init__(
        self,
        mammo_df: pd.DataFrame,
        us_df: pd.DataFrame,
        root: Path,
        transform=None,
        clinical_encoder: ClinicalEncoder | None = None,
        pair_seed: int = 0,
        match_on_label: bool = False,
    ):
        self.mammo = mammo_df.reset_index(drop=True)
        self.us = us_df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.enc = clinical_encoder
        self.match_on_label = match_on_label
        self.mammo_dataset = BreastDataset(self.mammo, self.root, transform)
        self.us_dataset = BreastDataset(self.us, self.root, transform)
        self.pairs = self._build_pairs(pair_seed)

    def _build_pairs(self, seed: int) -> list[tuple[int, int]]:
        if self.mammo.empty or self.us.empty:
            raise ValueError("Impossible de créer des paires avec un dataset vide.")

        rng = np.random.default_rng(seed)
        pairs: list[tuple[int, int]] = []

        if not self.match_on_label:
            # Test d'intégrité : appariement aveugle au label.
            us_idx = rng.permutation(len(self.us))
            return [(i, int(us_idx[i % len(us_idx)])) for i in range(len(self.mammo))]

        for label in (0, 1):
            m_idx = self.mammo.index[self.mammo["label"] == label].to_numpy()
            u_idx = self.us.index[self.us["label"] == label].to_numpy()
            if len(m_idx) == 0 or len(u_idx) == 0:
                continue
            # Échantillonnage avec remise côté échographie : les datasets
            # échographiques sont beaucoup plus petits que les mammographiques.
            chosen = rng.choice(u_idx, size=len(m_idx), replace=True)
            pairs += [(int(m), int(u)) for m, u in zip(m_idx, chosen)]

        if not pairs:
            raise ValueError("Aucune paire disponible : vérifiez les labels des deux datasets.")
        pairs.sort(key=lambda pair: pair[0])
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        m_i, u_i = self.pairs[idx]
        m_row, u_row = self.mammo.iloc[m_i], self.us.iloc[u_i]

        sample = {
            "mammo": self.mammo_dataset[m_i]["image"],
            "us": self.us_dataset[u_i]["image"],
            "label": torch.tensor(float(m_row["label"]), dtype=torch.float32),
            "mammo_index": m_i,
            "us_index": u_i,
        }
        if self.enc is not None:
            # Les variables cliniques suivent la mammographie : c'est la
            # modalité dont on dispose réellement de l'âge (CMMD).
            sample["clinical"] = torch.from_numpy(self.enc.transform_row(m_row))
        return sample
