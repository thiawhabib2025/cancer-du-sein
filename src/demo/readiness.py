"""Fail-closed helpers for exposing model predictions in the demo."""

from __future__ import annotations

import importlib.util
import math
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_IMAGE_BYTES = 20_000_000
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "TIFF"}


def inspect_runtime(project_root: Path) -> dict[str, object]:
    """Report prerequisites without treating file presence as model validation."""
    checkpoint_dir = Path(project_root) / "checkpoints"
    checkpoints = (
        sorted(
            path.name
            for path in checkpoint_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
        )
        if checkpoint_dir.is_dir()
        else []
    )
    return {
        "checkpoints": checkpoints,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "timm_available": importlib.util.find_spec("timm") is not None,
        "manifest_available": (checkpoint_dir / "model_manifest.json").is_file(),
        "inference_enabled": False,
    }


def decode_image_preview(
    content: bytes,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[Image.Image, dict[str, object]]:
    """Validate an uploaded image and return a display-only RGB preview."""
    if not content:
        raise ValueError("Le fichier image est vide.")
    if len(content) > max_bytes:
        raise ValueError(
            f"Fichier trop volumineux : limite de {max_bytes / 1_000_000:.0f} Mo."
        )

    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            width, height = image.size
            input_mode = image.mode
            frame_count = getattr(image, "n_frames", 1)
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise ValueError(f"Format non pris en charge. Formats acceptés : {supported}.")
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError(
                    f"Dimensions non prises en charge : {width} × {height} pixels."
                )
            image.verify()

        with Image.open(BytesIO(content)) as image:
            image.seek(0)
            preview = ImageOps.exif_transpose(image).convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Le fichier est illisible ou n'est pas une image valide.") from exc

    metadata: dict[str, object] = {
        "format": image_format,
        "width": width,
        "height": height,
        "mode": input_mode,
        "frames": frame_count,
    }
    return preview, metadata


def predict_probability(
    predictor: Callable[[object], object] | None,
    image: object,
) -> float:
    """Return a finite scalar probability only after a real predictor succeeds."""
    if predictor is None or not callable(predictor):
        raise RuntimeError("Aucun modèle vérifié et chargé pour l'inférence.")

    raw_score = predictor(image)
    if isinstance(raw_score, bool):
        raise ValueError("Le modèle doit retourner une probabilité numérique.")
    try:
        score = float(raw_score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Le modèle n'a pas retourné une probabilité scalaire.") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("La sortie du modèle n'est pas une probabilité finie entre 0 et 1.")
    return score
