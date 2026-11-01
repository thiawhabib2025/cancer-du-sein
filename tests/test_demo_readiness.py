from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from src.demo.readiness import (
    decode_image_preview,
    inspect_runtime,
    predict_probability,
)


def test_runtime_does_not_treat_checkpoint_presence_as_validation(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_best.pt").write_bytes(b"not-a-validated-checkpoint")

    status = inspect_runtime(tmp_path)

    assert status["checkpoints"] == ["model_best.pt"]
    assert status["inference_enabled"] is False


def test_prediction_requires_a_real_callable():
    with pytest.raises(RuntimeError, match="Aucun modèle vérifié"):
        predict_probability(None, object())


def test_prediction_returns_only_a_valid_probability():
    assert predict_probability(lambda _: 0.73, object()) == 0.73

    for output in (float("nan"), float("inf"), -0.1, 1.1, "invalid", True):
        with pytest.raises(ValueError):
            predict_probability(lambda _, value=output: value, object())


def test_predictor_failure_is_not_converted_to_a_score():
    def failing_predictor(_):
        raise RuntimeError("inférence indisponible")

    with pytest.raises(RuntimeError, match="inférence indisponible"):
        predict_probability(failing_predictor, object())


def test_image_preview_validates_and_normalizes_image():
    buffer = BytesIO()
    Image.new("L", (24, 16), color=128).save(buffer, format="PNG")

    preview, metadata = decode_image_preview(buffer.getvalue())

    assert preview.mode == "RGB"
    assert preview.size == (24, 16)
    assert metadata == {
        "format": "PNG",
        "width": 24,
        "height": 16,
        "mode": "L",
        "frames": 1,
    }


@pytest.mark.parametrize("image_format", ["JPEG", "TIFF"])
def test_image_preview_accepts_other_documented_formats(image_format):
    buffer = BytesIO()
    Image.new("RGB", (10, 8), color="white").save(buffer, format=image_format)

    preview, metadata = decode_image_preview(buffer.getvalue())

    assert preview.size == (10, 8)
    assert metadata["format"] == image_format


@pytest.mark.parametrize(
    ("content", "kwargs", "message"),
    [
        (b"", {}, "vide"),
        (b"not an image", {}, "illisible"),
        (b"too large", {"max_bytes": 4}, "volumineux"),
    ],
)
def test_image_preview_rejects_invalid_or_oversized_content(content, kwargs, message):
    with pytest.raises(ValueError, match=message):
        decode_image_preview(content, **kwargs)


def test_image_preview_rejects_unsupported_format():
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="GIF")

    with pytest.raises(ValueError, match="Format non pris en charge"):
        decode_image_preview(buffer.getvalue())


def test_image_preview_rejects_excessive_dimensions():
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")

    with pytest.raises(ValueError, match="Dimensions non prises en charge"):
        decode_image_preview(buffer.getvalue(), max_pixels=32)


def test_streamlit_demo_renders_without_claiming_a_prediction():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "demo_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not app.exception
    assert any("Inférence indisponible" in item.value for item in app.error)
    assert any("Aucune inférence n’est exécutée" in item.value for item in app.info)
    assert any("non validé cliniquement" in item.value for item in app.markdown)
    assert len(app.get("file_uploader")) == 2
    assert [tab.label for tab in app.tabs] == [
        "Démonstration visuelle",
        "Méthodologie & limites",
        "État technique",
    ]
    assert any("Effacer les images" == item.label for item in app.button)
    app.button[0].click().run(timeout=10)
    assert not app.exception
    assert app.session_state["upload_revision"] == 1
