"""
Boucle d'entraînement générique.

Trois exigences dictent la conception, toutes issues de contraintes réelles du
projet plutôt que d'un souci d'élégance.

**Colab coupe les sessions sans prévenir.** Perdre quatre heures d'entraînement
arrive à toute équipe une fois. Le checkpointing est donc systématique et écrit
sur Drive après chaque amélioration, avec reprise possible (`resume`).

**Le projet enchaîne une centaine d'entraînements.** Sans traçabilité, personne
ne saura en novembre quel chiffre correspond à quelle configuration. Chaque
exécution ajoute une ligne à `results/runs.csv` avec le hash du commit, le nom
de la config et le seed.

**La sensibilité prime sur l'accuracy.** Le critère de sélection du meilleur
modèle est l'AUC, pas l'accuracy — cette dernière est trompeuse en classes
déséquilibrées et sans rapport avec l'usage clinique. Quand les identifiants
patients sont fournis, cette AUC est calculée après agrégation par patiente.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch, nn = None, None

from src.eval.metrics import aggregate_patient_scores, compute_metrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    patience: int = 8                 # early stopping
    amp: bool = True                  # mixed precision : ~2x plus rapide
    grad_clip: float = 1.0
    monitor: str = "auc_roc"          # jamais 'accuracy'
    seed: int = 42
    warmup_epochs: int = 2
    pos_weight: float | None = None   # None → calculé depuis le train
    run_name: str = "run"
    out_dir: Path = field(default=Path("checkpoints"))


def _git_hash() -> str:
    """Hash du dépôt du projet, pas d'un dépôt parent de l'environnement."""
    project_root = Path(__file__).resolve().parents[2]
    try:
        git_root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()).resolve()
        if git_root != project_root.resolve():
            return "nogit"
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Déterminisme complet : coûte ~10 % de vitesse, mais un résultat
        # non reproductible n'est pas défendable en soutenance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Entraîneur
# ---------------------------------------------------------------------------
class Trainer:
    """Entraîne un modèle binaire et suit les métriques par époque."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg: TrainConfig,
        device: str | None = None,
        forward_fn=None,
        validation_groups=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # `forward_fn` permet de réutiliser cette boucle pour la fusion, où le
        # batch contient deux images plus les variables cliniques.
        self.forward_fn = forward_fn or self._default_forward
        self.validation_groups = None
        if validation_groups is not None:
            groups = np.asarray(validation_groups, dtype=object)
            if groups.ndim != 1 or len(groups) == 0 or any(
                value is None
                or (isinstance(value, (float, np.floating)) and np.isnan(value))
                for value in groups
            ):
                raise ValueError("validation_groups doit contenir des identifiants patients valides.")
            if len(groups) != len(self.val_loader.dataset):
                raise ValueError(
                    "validation_groups doit contenir un identifiant par exemple "
                    "dans le DataLoader de validation."
                )
            self.validation_groups = groups
            self.validation_group_hash = hashlib.sha256(
                json.dumps(groups.tolist(), default=str).encode()
            ).hexdigest()
        else:
            self.validation_group_hash = None

        set_seed(cfg.seed)

        pw = cfg.pos_weight
        if pw is None:
            pw = self._estimate_pos_weight()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], device=self.device) if pw else None
        )

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs)
        )
        # torch>=2.4 : torch.cuda.amp.* est déprécié au profit de torch.amp.*
        _amp_on = cfg.amp and self.device == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=_amp_on)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=_amp_on)
        self._amp_on = _amp_on

        self.best_score = -np.inf
        self.best_epoch = -1
        self.best_model_state: dict | None = None
        self.history: list[dict] = []
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _estimate_pos_weight(self) -> float | None:
        """Pondère la classe positive selon le déséquilibre observé.

        Sur CBIS-DDSM le déséquilibre est modéré ; sur BUSI, les malignes sont
        minoritaires. Sans pondération, le modèle privilégie la spécificité au
        détriment de la sensibilité — exactement l'inverse de ce qu'on veut
        cliniquement.
        """
        labels = []
        for batch in self.train_loader:
            if "label" not in batch:
                raise KeyError("Chaque batch d'entraînement doit contenir 'label'.")
            batch_labels = batch["label"]
            if torch.is_tensor(batch_labels):
                batch_labels = batch_labels.detach().cpu().numpy()
            labels.append(np.asarray(batch_labels).ravel())
        if not labels:
            raise ValueError("Le DataLoader d'entraînement ne contient aucun batch.")
        labels = np.concatenate(labels)
        n_pos = float(labels.sum())
        n_neg = float(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None
        w = n_neg / n_pos
        return float(np.clip(w, 0.5, 10.0))   # borné : au-delà, l'entraînement diverge

    def _default_forward(self, batch):
        image = batch["image"].to(self.device, non_blocking=True)
        if "clinical" in batch:
            return self.model(image, batch["clinical"].to(self.device))
        return self.model(image)

    # ------------------------------------------------------------------
    def _run_epoch(self, loader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train() if train else self.model.eval()
        losses, scores, targets = [], [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                y = batch["label"].to(self.device).float()

                with torch.amp.autocast("cuda", enabled=self._amp_on):
                    logits = self.forward_fn(batch)
                    loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.cfg.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                losses.append(loss.item())
                scores.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
                targets.append(y.detach().cpu().numpy())

        return float(np.mean(losses)), np.concatenate(scores), np.concatenate(targets)

    # ------------------------------------------------------------------
    def _checkpoint_config(self) -> dict:
        config = asdict(self.cfg)
        config["out_dir"] = str(config["out_dir"])
        config["validation_group_hash"] = self.validation_group_hash
        return config

    def _validation_metrics(self, scores: np.ndarray, targets: np.ndarray) -> dict:
        if self.validation_groups is not None:
            targets, scores, _ = aggregate_patient_scores(
                targets, scores, self.validation_groups
            )
        return compute_metrics(targets, scores)

    @staticmethod
    def _atomic_torch_save(payload: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json_save(payload: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
            )
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _rng_state(self) -> dict:
        state = {
            "python": __import__("random").getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        import random

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _resume(self, path: Path, fingerprint: str | None) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        expected_config = self._checkpoint_config()
        saved_config = state.get("config")
        if saved_config != expected_config:
            raise ValueError(
                f"Configuration incompatible avec le checkpoint {path}; "
                "utilisez un autre run_name ou restaurez les paramètres d'origine."
            )
        if state.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Empreinte de données/protocole incompatible avec le checkpoint {path}."
            )

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.best_model_state = state["best_model_state"]
        self.history = state["history"]
        self._restore_rng_state(state["rng_state"])
        start_epoch = int(state["epoch"]) + 1
        print(f"  Reprise de {path.name} à l’époque {start_epoch}")
        return start_epoch

    def fit(
        self,
        resume: bool = False,
        fingerprint: str | None = None,
    ) -> dict:
        cfg = self.cfg
        ckpt = cfg.out_dir / f"{cfg.run_name}_best.pt"
        last_ckpt = cfg.out_dir / f"{cfg.run_name}_last.pt"
        history_path = cfg.out_dir / f"{cfg.run_name}_history.json"
        start = time.time()
        start_epoch = 0

        if resume and last_ckpt.exists():
            start_epoch = self._resume(last_ckpt, fingerprint)

        for epoch in range(start_epoch, cfg.epochs):
            tr_loss, _, _ = self._run_epoch(self.train_loader, train=True)
            va_loss, va_scores, va_targets = self._run_epoch(self.val_loader, train=False)

            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            metrics = self._validation_metrics(va_scores, va_targets)

            record = {
                "epoch": epoch,
                "train_loss": round(tr_loss, 4),
                "val_loss": round(va_loss, 4),
                "lr": round(self.optimizer.param_groups[0]["lr"], 6),
                **metrics,
            }
            self.history.append(record)

            if cfg.monitor not in metrics:
                raise KeyError(
                    f"Métrique de suivi '{cfg.monitor}' absente des résultats : "
                    f"{sorted(metrics)}"
                )
            score = metrics[cfg.monitor]
            improved = score > self.best_score
            if improved:
                self.best_score, self.best_epoch = score, epoch
                self.best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "best_model_state": self.best_model_state,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "history": self.history,
                "config": self._checkpoint_config(),
                "fingerprint": fingerprint,
                "rng_state": self._rng_state(),
                "git": _git_hash(),
            }
            self._atomic_torch_save(checkpoint, last_ckpt)
            self._atomic_json_save(
                {
                    "run_name": cfg.run_name,
                    "config": self._checkpoint_config(),
                    "fingerprint": fingerprint,
                    "best_epoch": self.best_epoch,
                    "best_score": self.best_score,
                    "history": self.history,
                    "checkpoint": str(ckpt),
                    "last_checkpoint": str(last_ckpt),
                },
                history_path,
            )

            print(
                f"  ép.{epoch:>3} | train {tr_loss:.4f} | val {va_loss:.4f} | "
                f"AUC {metrics.get('auc_roc', 0):.4f} | "
                f"sens {metrics.get('sensitivity', 0):.3f} | "
                f"spéc {metrics.get('specificity', 0):.3f}"
                + ("  ★" if improved else "")
            )

            if epoch - self.best_epoch >= cfg.patience:
                print(f"  ⏹ Early stopping (aucune amélioration depuis {cfg.patience} époques)")
                break

        elapsed = time.time() - start
        if self.best_model_state is None:
            raise RuntimeError(
                f"Aucun checkpoint valide n'a été produit pour {cfg.run_name}."
            )
        self._atomic_torch_save(
            {
                "epoch": self.best_epoch,
                "model_state": self.best_model_state,
                "score": self.best_score,
                "config": self._checkpoint_config(),
                "fingerprint": fingerprint,
                "git": _git_hash(),
            },
            ckpt,
        )
        print(f"\n  Meilleur {cfg.monitor} : {self.best_score:.4f} (époque {self.best_epoch})")
        print(f"  Durée : {elapsed / 60:.1f} min · checkpoint : {ckpt}")

        return {
            "run_name": cfg.run_name,
            "best_score": round(self.best_score, 4),
            "best_epoch": self.best_epoch,
            "epochs_run": len(self.history),
            "minutes": round(elapsed / 60, 2),
            "checkpoint": str(ckpt),
            "last_checkpoint": str(last_ckpt),
            "history": str(history_path),
            "git": _git_hash(),
        }

    # ------------------------------------------------------------------
    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Scores et cibles sur un loader — pour l'évaluation finale et DeLong."""
        _, scores, targets = self._run_epoch(loader, train=False)
        return scores, targets

    def load_best(self, path: Path | None = None) -> None:
        path = path or self.cfg.out_dir / f"{self.cfg.run_name}_best.pt"
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        print(f"  Chargé : {path} (époque {state['epoch']}, score {state['score']:.4f})")


# ---------------------------------------------------------------------------
# Journalisation des exécutions
# ---------------------------------------------------------------------------
def log_run(
    results_dir: Path,
    run_name: str,
    config: dict,
    metrics: dict,
    extra: dict | None = None,
) -> None:
    """Ajoute une ligne à `results/runs.csv`.

    Sans cette trace, la centaine d'entraînements du projet devient
    ingérable : impossible de savoir quel chiffre du rapport correspond à
    quelle configuration, ni de reproduire un résultat trois semaines plus tard.
    """
    import pandas as pd

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "runs.csv"

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "git": _git_hash(),
        "config": json.dumps(config, default=str, ensure_ascii=False),
        **metrics,
        **(extra or {}),
    }
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"  Journalisé dans {path}")
