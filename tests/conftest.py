"""Fixtures partagées — manifeste factice imitant la structure réelle.

Permet aux tests critiques de tourner sans avoir téléchargé les datasets,
donc dès le premier `git clone` et à chaque commit en CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Rend `src` importable sans installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def fake_manifest() -> pd.DataFrame:
    """Manifeste factice de 600 patientes, avec plusieurs images par patiente.

    Reproduit la caractéristique piégeuse des données réelles : CBIS-DDSM a des
    vues CC et MLO pour la même patiente, BUSI plusieurs images pour 600
    patientes réparties sur 780 fichiers. C'est exactement ce qui provoque la
    fuite si le découpage se fait au niveau image.

    600 patientes et non 200 : en dessous, l'erreur d'échantillonnage sur l'AUC
    (SE ≈ 0,04 à n=200) rend l'audit anti-fuite instable d'une exécution à
    l'autre. Un garde-fou qui clignote au hasard finit par être désactivé.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(600):
        pid = f"cbis_P{i:05d}"
        label = int(rng.random() < 0.4)
        n_images = int(rng.integers(1, 4))  # 1 à 3 images par patiente
        for j, view in enumerate(["CC", "MLO", "CC"][:n_images]):
            rows.append(
                {
                    "patient_id": pid,
                    "image_path": f"cbis_ddsm/{pid}_{view}_{j}.png",
                    "dataset": "cbis_ddsm",
                    "modality": "mammography",
                    "view": view,
                    "laterality": "L" if j % 2 == 0 else "R",
                    "label": label,
                    "lesion_type": "mass",
                    "age": np.nan,
                    "density": "",
                    "mask_path": "",
                }
            )
    return pd.DataFrame(rows)
