"""Tests for deterministic virtual pairing and patient-level evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.datasets import PairedDataset
from src.eval.metrics import aggregate_patient_scores
from src.train import trainer


def _manifest(prefix: str, labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"{prefix}_{i}" for i in range(len(labels))],
            "image_path": [f"{prefix}_{i}.png" for i in range(len(labels))],
            "label": labels,
        }
    )


def test_virtual_label_matched_pairs_are_reproducible_and_ordered():
    mammography = _manifest("mammo", [1, 0, 1, 0, 1, 0])
    ultrasound = _manifest("us", [0, 1, 0, 1])

    first = PairedDataset(mammography, ultrasound, root=".", match_on_label=True)
    repeated = PairedDataset(mammography, ultrasound, root=".", match_on_label=True)

    assert first.pairs == repeated.pairs
    assert [m for m, _ in first.pairs] == list(range(len(mammography)))
    assert all(
        mammography.iloc[m].label == ultrasound.iloc[u].label
        for m, u in first.pairs
    )


def test_default_pairing_ignores_labels():
    mammography = _manifest("mammo", [1, 0, 1, 0])
    ultrasound = _manifest("us", [0, 0, 0, 0])
    paired = PairedDataset(mammography, ultrasound, root=".", pair_seed=7)

    assert [m for m, _ in paired.pairs] == list(range(len(mammography)))
    assert any(
        mammography.iloc[m].label != ultrasound.iloc[u].label
        for m, u in paired.pairs
    )


def test_random_pairing_preserves_mammography_reference_order():
    mammography = _manifest("mammo", [1, 0, 1, 0])
    ultrasound = _manifest("us", [0, 0, 1])
    paired = PairedDataset(
        mammography, ultrasound, root=".", pair_seed=7, match_on_label=False
    )

    assert [m for m, _ in paired.pairs] == list(range(len(mammography)))


def test_pairing_empty_datasets_fails_explicitly():
    with pytest.raises(ValueError, match="dataset vide"):
        PairedDataset(
            _manifest("mammo", [0, 1]),
            _manifest("us", []),
            root=".",
        )


def test_patient_aggregation_uses_any_positive_and_mean_score():
    y, scores, patients = aggregate_patient_scores(
        [0, 1, 0, 1, 1],
        [0.2, 0.8, 0.3, 0.6, 0.9],
        ["p2", "p2", "p1", "p3", "p3"],
    )

    assert patients.tolist() == ["p1", "p2", "p3"]
    assert y.tolist() == [0, 1, 1]
    np.testing.assert_allclose(scores, [0.3, 0.5, 0.75])


def test_patient_aggregation_can_use_max_score():
    _, scores, _ = aggregate_patient_scores(
        [0, 1, 0], [0.2, 0.8, 0.4], ["p1", "p1", "p2"], score_aggregation="max"
    )
    np.testing.assert_allclose(scores, [0.8, 0.4])


@pytest.mark.parametrize(
    ("y", "scores", "patients", "message"),
    [
        ([0], [0.1, 0.2], ["p1"], "même longueur"),
        ([2], [0.1], ["p1"], "binaire"),
        ([0], [np.nan], ["p1"], "non finies"),
        ([0], [0.1], [None], "valeurs manquantes"),
    ],
)
def test_patient_aggregation_rejects_invalid_inputs(y, scores, patients, message):
    with pytest.raises(ValueError, match=message):
        aggregate_patient_scores(y, scores, patients)


def test_git_hash_does_not_use_a_parent_environment_repository(monkeypatch, tmp_path):
    def fake_check_output(args, **kwargs):
        assert args == ["git", "rev-parse", "--show-toplevel"]
        return str(tmp_path).encode()

    monkeypatch.setattr(trainer.subprocess, "check_output", fake_check_output)

    assert trainer._git_hash() == "nogit"
