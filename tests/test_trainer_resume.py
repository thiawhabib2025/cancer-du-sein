import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch requis pour tester la reprise")

from torch.utils.data import DataLoader, TensorDataset

from src.train.trainer import Trainer, TrainConfig


class BatchDataset(TensorDataset):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return {"image": image, "label": label}


def _trainer(tmp_path, model, config, validation_groups=None):
    features = torch.tensor(
        [[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.2], [2.0, 0.0]] * 2
    )
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0] * 2)
    train = DataLoader(BatchDataset(features, labels), batch_size=4, shuffle=True)
    validation = DataLoader(BatchDataset(features, labels), batch_size=4)
    return Trainer(
        model,
        train,
        validation,
        config,
        device="cpu",
        forward_fn=lambda batch: model(batch["image"]).squeeze(1),
        validation_groups=validation_groups,
    )


def test_validation_metrics_are_aggregated_by_patient(tmp_path):
    config = TrainConfig(
        epochs=1,
        amp=False,
        run_name="grouped-validation",
        out_dir=tmp_path,
    )
    groups = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
    trainer = _trainer(tmp_path, torch.nn.Linear(2, 1), config, groups)
    scores = np.asarray([0.1, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8])
    targets = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])

    metrics = trainer._validation_metrics(scores, targets)

    assert metrics["n"] == 4
    assert metrics["n_positive"] == 2


def test_fit_resumes_from_last_epoch_checkpoint(tmp_path):
    config = TrainConfig(
        epochs=2,
        patience=5,
        warmup_epochs=0,
        amp=False,
        run_name="resume-test",
        out_dir=tmp_path,
    )
    first = _trainer(tmp_path, torch.nn.Linear(2, 1), config)
    original_run_epoch = first._run_epoch
    calls = 0

    def interrupt_after_one_epoch(loader, train):
        nonlocal calls
        if calls == 2:
            raise RuntimeError("simulated interruption")
        calls += 1
        return original_run_epoch(loader, train)

    first._run_epoch = interrupt_after_one_epoch
    with pytest.raises(RuntimeError, match="simulated interruption"):
        first.fit(fingerprint="protocol-v1")

    last_checkpoint = tmp_path / "resume-test_last.pt"
    history = tmp_path / "resume-test_history.json"
    assert last_checkpoint.exists()
    assert history.exists()

    resumed = _trainer(tmp_path, torch.nn.Linear(2, 1), config)
    result = resumed.fit(resume=True, fingerprint="protocol-v1")

    assert [row["epoch"] for row in resumed.history] == [0, 1]
    assert result["best_epoch"] >= 0
    assert (tmp_path / "resume-test_best.pt").exists()


def test_resume_rejects_different_protocol_fingerprint(tmp_path):
    config = TrainConfig(
        epochs=1,
        patience=2,
        warmup_epochs=0,
        amp=False,
        run_name="fingerprint-test",
        out_dir=tmp_path,
    )
    trainer = _trainer(tmp_path, torch.nn.Linear(2, 1), config)
    trainer.fit(fingerprint="protocol-v1")

    resumed = _trainer(tmp_path, torch.nn.Linear(2, 1), config)
    with pytest.raises(ValueError, match="Empreinte"):
        resumed.fit(resume=True, fingerprint="protocol-v2")
