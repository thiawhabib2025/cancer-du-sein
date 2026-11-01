from __future__ import annotations

import pandas as pd
import yaml

from src.data import build_manifest


def test_build_normalizes_windows_image_and_mask_paths(tmp_path, monkeypatch):
    raw_dir = tmp_path / "busi"
    raw_dir.mkdir()
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"datasets": {"busi": {"enabled": True, "raw_dir": str(raw_dir)}}}
        ),
        encoding="utf-8",
    )
    rows = pd.DataFrame(
        [
            {
                "patient_id": "busi_case1",
                "image_path": r"benign\case1.png",
                "dataset": "busi",
                "modality": "ultrasound",
                "view": "US",
                "laterality": "unknown",
                "label": 0,
                "lesion_type": "unknown",
                "age": float("nan"),
                "density": "",
                "mask_path": r"benign\case1_mask.png",
                "clinical_level": "C0",
            },
            {
                "patient_id": "busi_case2",
                "image_path": r"benign\case2.png",
                "dataset": "busi",
                "modality": "ultrasound",
                "view": "US",
                "laterality": "unknown",
                "label": 0,
                "lesion_type": "unknown",
                "age": float("nan"),
                "density": "",
                "mask_path": float("nan"),
                "clinical_level": "C0",
            },
        ]
    )
    monkeypatch.setitem(build_manifest._PARSERS, "busi", lambda cfg, path: rows)

    manifest = build_manifest.build(config_path)

    assert manifest.loc[0, "image_path"] == "busi/benign/case1.png"
    assert manifest.loc[0, "mask_path"] == "busi/benign/case1_mask.png"
    assert manifest.loc[1, "image_path"] == "busi/benign/case2.png"
    assert manifest.loc[1, "mask_path"] == ""
