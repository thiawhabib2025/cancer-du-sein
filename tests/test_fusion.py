"""Tests structurels des modules de fusion.

Nécessitent torch — ignorés proprement s'il n'est pas installé (l'environnement
de développement n'a pas de GPU ; ces tests tournent sur Colab avant tout
entraînement).

Ils vérifient les invariants qui coûteraient des heures de GPU s'ils étaient
faux : formes de sortie, gradients qui circulent vers les trois encodeurs,
poids de modalité qui somment à 1, registre complet.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch requis (exécuter sur Colab)")

from src.models.fusion import (   # noqa: E402
    FUSION_REGISTRY,
    LogitAveraging,
    build_fusion,
)

B, C, H, W = 2, 3, 224, 224
CLIN_DIM = 8


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(0)
    return {
        "mammo": torch.randn(B, C, H, W, generator=g),
        "us": torch.randn(B, C, H, W, generator=g),
        "clinical": torch.randn(B, CLIN_DIM, generator=g),
    }


def _make(name):
    return build_fusion(
        name, encoder_name="resnet18", pretrained=False,
        clinical_dim=CLIN_DIM, image_size=224,
    )


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(FUSION_REGISTRY))
def test_output_shape_is_logit_per_sample(name, batch):
    model = _make(name).eval()
    with torch.no_grad():
        out = model(batch["mammo"], batch["us"], batch["clinical"])
    assert out.shape == (B,), f"{name} : attendu ({B},), obtenu {tuple(out.shape)}"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", list(FUSION_REGISTRY))
def test_gradients_reach_all_encoders(name, batch):
    """Un module de fusion mal branché peut ignorer silencieusement une
    modalité — le modèle s'entraîne, converge, et personne ne voit que l'écho
    n'a jamais servi. On vérifie que le gradient atteint chaque encodeur."""
    model = _make(name).train()
    out = model(batch["mammo"], batch["us"], batch["clinical"])
    out.sum().backward()

    def has_grad(module):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )

    if name == "early":
        assert has_grad(model.encoder), "early : encodeur sans gradient"
        assert has_grad(model.clinical_encoder), "early : clinique sans gradient"
    else:
        assert has_grad(model.mammo_encoder), f"{name} : encodeur mammo sans gradient"
        assert has_grad(model.us_encoder), f"{name} : encodeur écho sans gradient"
        assert has_grad(model.clinical_encoder), f"{name} : encodeur clinique sans gradient"


def test_logit_averaging_weights_sum_to_one(batch):
    model = _make("logit_averaging")
    w = model.modality_weights()
    assert set(w) == {"mammo", "us", "clinical"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # À l'initialisation (poids nuls → softmax uniforme) : ~1/3 chacun.
    assert all(abs(v - 1 / 3) < 1e-6 for v in w.values())


def test_amw_exposes_per_case_weights(batch):
    model = _make("cross_attention_amw").eval()
    with torch.no_grad():
        model(batch["mammo"], batch["us"], batch["clinical"])
    w = model.last_modality_weights
    assert w is not None and w.shape == (B, 3)
    assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-5)
    assert (w >= 0).all()


def test_early_fusion_first_conv_takes_6_channels(batch):
    model = _make("early")
    first = next(m for m in model.encoder.backbone.modules()
                 if isinstance(m, torch.nn.Conv2d))
    assert first.in_channels == 6


def test_works_without_clinical():
    """clinical_dim=0 : les modèles doivent fonctionner en bi-modal pur
    (nécessaire pour l'ablation « clinique absente » du protocole)."""
    g = torch.Generator().manual_seed(1)
    m = torch.randn(B, C, H, W, generator=g)
    u = torch.randn(B, C, H, W, generator=g)
    for name in FUSION_REGISTRY:
        model = build_fusion(name, encoder_name="resnet18",
                             pretrained=False, clinical_dim=0)
        out = model(m, u)
        assert out.shape == (B,), f"{name} sans clinique : forme {tuple(out.shape)}"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="inconnue"):
        build_fusion("attention_quantique")


def test_missing_clinical_raises(batch):
    model = _make("late")
    with pytest.raises(ValueError, match="cliniques"):
        model(batch["mammo"], batch["us"], None)
