"""
Conversion DICOM → PNG 16 bits.

POURQUOI CE MODULE N'EST PAS TRIVIAL
------------------------------------
La tentation est d'écrire trois lignes :

    arr = pydicom.dcmread(path).pixel_array
    cv2.imwrite(out, (arr / arr.max() * 255).astype(np.uint8))

C'est faux, et de trois façons différentes.

1. **Les LUT DICOM sont ignorées.** Un fichier DICOM porte une table de
   correspondance (VOI LUT / fenêtrage) qui indique comment mapper les valeurs
   brutes du capteur vers une échelle visuelle. Sans elle, le contraste local
   s'écrase et les **microcalcifications disparaissent** — or ce sont
   précisément les signes précoces que le projet cherche à détecter.

2. **MONOCHROME1 n'est pas géré.** Certaines mammographies sont stockées en
   photométrie inversée (valeurs hautes = noir). Sans inversion, une partie du
   dataset se retrouve en négatif. Le réseau apprend alors surtout à distinguer
   les deux conventions de stockage.

3. **Le passage en 8 bits détruit de l'information.** Les mammographies sont
   acquises en 12–16 bits. Écraser en 8 bits dès le prétraitement supprime des
   nuances de densité qu'on ne récupérera jamais. On écrit donc en PNG 16 bits
   et on ne convertit qu'au moment du chargement.

Un quatrième point, moins grave mais coûteux : les images brutes contiennent
50 à 70 % de fond noir. Le recadrage sur la région mammaire améliore la
résolution effective à taille d'entrée constante.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import yaml

__all__ = ["dicom_to_array", "crop_to_breast", "convert_file", "convert_dataset"]


# ---------------------------------------------------------------------------
# Lecture DICOM
# ---------------------------------------------------------------------------
def dicom_to_array(path: Path, apply_lut: bool = True) -> np.ndarray:
    """Lit un DICOM et retourne un tableau uint16 correctement orienté.

    Applique, dans l'ordre :
      1. la VOI LUT (ou le fenêtrage window/level à défaut) ;
      2. l'inversion si PhotometricInterpretation == MONOCHROME1 ;
      3. la normalisation vers la pleine plage 16 bits.

    Returns
    -------
    np.ndarray, dtype uint16, shape (H, W)
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array

    if apply_lut:
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception as exc:               # LUT absente ou malformée
            warnings.warn(f"{path.name} : VOI LUT non applicable ({exc}) — valeurs brutes.")

    # MONOCHROME1 : valeurs hautes = noir → inverser
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = arr.max() - arr

    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        warnings.warn(f"{path.name} : image constante.")
        return np.zeros(arr.shape, dtype=np.uint16)

    return ((arr - lo) / (hi - lo) * 65535.0).astype(np.uint16)


# ---------------------------------------------------------------------------
# Recadrage sur la région mammaire
# ---------------------------------------------------------------------------
def crop_to_breast(
    arr: np.ndarray,
    threshold_ratio: float = 0.05,
    margin: int = 10,
) -> np.ndarray:
    """Recadre sur la plus grande composante connexe non noire.

    Les mammographies brutes sont majoritairement du fond. À taille d'entrée
    constante (224×224), recadrer double environ la résolution effective du
    tissu — un gain gratuit et non négligeable.

    Prudence intégrée : si la boîte détectée est aberrante (moins de 10 % de
    l'image, ce qui trahit un artefact ou une étiquette métallique captée à la
    place du sein), on renvoie l'image intacte plutôt qu'un recadrage faux.
    """
    import cv2

    a8 = (arr.astype(np.float32) / 257.0).astype(np.uint8)      # 16 → 8 bits
    thresh = max(1, int(255 * threshold_ratio))
    _, binary = cv2.threshold(a8, thresh, 255, cv2.THRESH_BINARY)

    # Ferme les petits trous pour éviter de fragmenter le sein.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    if (w * h) < 0.10 * arr.size:
        warnings.warn("Recadrage suspect (< 10 % de l'image) — image conservée intacte.")
        return arr

    H, W = arr.shape
    y0, y1 = max(0, y - margin), min(H, y + h + margin)
    x0, x1 = max(0, x - margin), min(W, x + w + margin)
    return arr[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert_file(
    src: Path,
    dst: Path,
    apply_lut: bool = True,
    crop: bool = True,
) -> bool:
    """Convertit un DICOM en PNG 16 bits. Retourne False en cas d'échec."""
    import cv2

    try:
        arr = dicom_to_array(src, apply_lut=apply_lut)
        if crop:
            arr = crop_to_breast(arr)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # PNG 16 bits : OpenCV le gère nativement pour un uint16 mono-canal.
        return bool(cv2.imwrite(str(dst), arr))
    except Exception as exc:
        warnings.warn(f"Échec sur {src} : {exc}")
        return False


def convert_dataset(
    raw_dir: Path,
    out_dir: Path,
    apply_lut: bool = True,
    crop: bool = True,
    limit: int | None = None,
) -> dict:
    """Convertit récursivement tous les DICOM d'un dossier.

    Conserve l'arborescence relative pour que les chemins du manifeste restent
    prévisibles.
    """
    files = sorted(list(raw_dir.rglob("*.dcm")) + list(raw_dir.rglob("*.DCM")))
    if limit:
        files = files[:limit]
    if not files:
        warnings.warn(f"Aucun DICOM sous {raw_dir}")
        return {"total": 0, "ok": 0, "failed": 0}

    ok = 0
    for i, src in enumerate(files, 1):
        dst = out_dir / src.relative_to(raw_dir).with_suffix(".png")
        ok += convert_file(src, dst, apply_lut, crop)
        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  ({ok} réussis)")

    stats = {"total": len(files), "ok": ok, "failed": len(files) - ok}
    if stats["failed"]:
        warnings.warn(f"{stats['failed']} fichier(s) en échec — inspectez les avertissements.")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Conversion DICOM → PNG 16 bits.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Convertir seulement N fichiers (test rapide).")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-lut", action="store_true",
                        help="DÉCONSEILLÉ : détruit le contraste des microcalcifications.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dcfg = cfg["datasets"].get(args.dataset)
    if dcfg is None:
        raise SystemExit(f"Dataset '{args.dataset}' absent de {args.config}")

    raw = Path(dcfg["raw_dir"])
    out = Path(cfg["paths"]["processed"]) / args.dataset

    if args.no_lut:
        warnings.warn(
            "--no-lut activé : le contraste des microcalcifications sera dégradé. "
            "À n'utiliser que pour une comparaison volontaire."
        )

    print(f"{raw}  →  {out}")
    stats = convert_dataset(
        raw, out,
        apply_lut=not args.no_lut,
        crop=not args.no_crop,
        limit=args.limit,
    )
    print(f"\n✅ {stats['ok']}/{stats['total']} convertis · {stats['failed']} échecs")


if __name__ == "__main__":
    main()
