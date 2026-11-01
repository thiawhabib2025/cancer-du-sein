"""
Stratégies de fusion multi-modale — le cœur scientifique du projet (CdC §6.3).

Cinq modèles, une même interface : (mammo, écho, clinique) → logit.

    LogitAveraging        ← LE BASELINE OBLIGATOIRE (décision d'équipe, piste B)
    EarlyFusion           a. concaténation précoce
    LateFusion            b. concaténation tardive     (Chen et al. 2025)
    CrossAttentionFusion  c. cross-attention           (Rashedi et al. 2026)
    CrossAttentionAMW     d. cross-attention + AMW     (Rashedi et al. 2026)

RAPPEL DU CONTEXTE — À LIRE AVANT D'INTERPRÉTER LES RÉSULTATS
-------------------------------------------------------------
Le projet suit la piste B : les paires mammo/écho sont des **patientes
virtuelles** (deux femmes différentes, appariées sans consulter leur label —
voir datasets.PairedDataset). La cible est le label mammographique ; il
n'existe donc AUCUNE interaction inter-modale réelle dans les données.

Conséquence : la cross-attention ne peut prétendre à mieux que la moyenne des
logits QUE si elle exploite autre chose que l'interaction (par exemple une
pondération adaptative de la qualité de chaque image). Le protocole complet est
dans docs/protocole_piste_B.md. La règle d'or :

    Toute expérience de fusion DOIT inclure LogitAveraging.
    Tout écart d'AUC DOIT passer par le test de DeLong.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch, nn, F = None, object, None

from src.models.encoders import ImageEncoder, ClinicalEncoderMLP

__all__ = [
    "LogitAveraging",
    "EarlyFusion",
    "LateFusion",
    "CrossAttentionFusion",
    "CrossAttentionAMW",
    "FUSION_REGISTRY",
]


# ---------------------------------------------------------------------------
# Base commune : trois encodeurs partagés par toutes les stratégies
# ---------------------------------------------------------------------------
class _FusionBase(nn.Module):
    """Encodeurs mammo + écho (+ clinique). Les stratégies héritent d'ici.

    Le protocole (CdC §6.3) impose les MÊMES extracteurs pour toutes les
    stratégies : seule la façon de combiner change. C'est ce qui rend la
    comparaison équitable.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        image_size: int = 224,
        clinical_dim: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mammo_encoder = ImageEncoder(encoder_name, pretrained, image_size)
        self.us_encoder = ImageEncoder(encoder_name, pretrained, image_size)
        self.embed_dim = self.mammo_encoder.out_dim

        self.clinical_encoder = None
        self.clinical_out = 0
        if clinical_dim > 0:
            self.clinical_encoder = ClinicalEncoderMLP(clinical_dim, dropout=dropout)
            self.clinical_out = self.clinical_encoder.out_dim

        self.dropout = dropout

    def _encode(self, mammo, us, clinical=None):
        f_m = self.mammo_encoder(mammo)
        f_u = self.us_encoder(us)
        f_c = None
        if self.clinical_encoder is not None:
            if clinical is None:
                raise ValueError("Le modèle attend des variables cliniques (C0).")
            f_c = self.clinical_encoder(clinical)
        return f_m, f_u, f_c


def _head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
    )


# ---------------------------------------------------------------------------
# BASELINE OBLIGATOIRE — moyenne des logits
# ---------------------------------------------------------------------------
class LogitAveraging(_FusionBase):
    """Moyenne (pondérée, apprise) des logits de têtes indépendantes.

    C'EST LE JUGE DE PAIX DE LA PISTE B. Chaque modalité produit son propre
    logit sans jamais voir les autres ; on combine à la toute fin par une somme
    pondérée dont les poids sont les seuls paramètres de fusion (3 scalaires).

    Aucune interaction inter-modale n'est modélisée — par construction. Si une
    stratégie « avancée » ne bat pas ce modèle au test de DeLong, son mécanisme
    d'interaction n'apporte rien sur nos données. Sur appariement virtuel,
    c'est l'issue la plus probable, et c'est un résultat en soi.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.head_m = _head(self.embed_dim, self.dropout)
        self.head_u = _head(self.embed_dim, self.dropout)
        self.head_c = _head(self.clinical_out, self.dropout) if self.clinical_out else None
        n = 3 if self.head_c is not None else 2
        self.weights = nn.Parameter(torch.zeros(n))    # softmax → moyenne au départ

    def forward(self, mammo, us, clinical=None):
        f_m, f_u, f_c = self._encode(mammo, us, clinical)
        logits = [self.head_m(f_m), self.head_u(f_u)]
        if self.head_c is not None:
            logits.append(self.head_c(f_c))
        w = torch.softmax(self.weights, dim=0)
        out = sum(wi * li for wi, li in zip(w, torch.stack(logits, 0).unbind(0)))
        return out.squeeze(1)

    def modality_weights(self) -> dict:
        """Poids appris — à reporter dans le rapport (quelle modalité domine ?)."""
        w = torch.softmax(self.weights.detach(), dim=0).cpu().numpy()
        names = ["mammo", "us"] + (["clinical"] if self.head_c is not None else [])
        return dict(zip(names, map(float, w)))


# ---------------------------------------------------------------------------
# a. Concaténation précoce
# ---------------------------------------------------------------------------
class EarlyFusion(nn.Module):
    """Fusion au niveau pixel : mammo et écho empilées en entrée d'un
    extracteur UNIQUE. Le clinique est concaténé après extraction.

    Inconvénient attendu (CdC §6.3.1) : les deux modalités sont mélangées dès
    la première convolution, chacune perd ses spécificités. Généralement la
    plus faible des quatre stratégies — elle est là comme borne basse.

    Implémentation : 6 canaux (2 × RGB). La première couche du backbone est
    remplacée pour accepter 6 canaux, ses poids pré-entraînés dupliqués puis
    divisés par 2 pour conserver l'échelle des activations.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        image_size: int = 224,
        clinical_dim: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ImageEncoder(encoder_name, pretrained, image_size)
        self._patch_first_conv()

        self.clinical_encoder = None
        clin_out = 0
        if clinical_dim > 0:
            self.clinical_encoder = ClinicalEncoderMLP(clinical_dim, dropout=dropout)
            clin_out = self.clinical_encoder.out_dim

        self.head = _head(self.encoder.out_dim + clin_out, dropout)

    def _patch_first_conv(self) -> None:
        """Passe la première convolution de 3 à 6 canaux d'entrée."""
        backbone = self.encoder.backbone
        # Trouve la première Conv2d (portable entre architectures timm).
        name, first = next(
            (n, m) for n, m in backbone.named_modules() if isinstance(m, nn.Conv2d)
        )
        new = nn.Conv2d(
            6, first.out_channels, first.kernel_size, first.stride,
            first.padding, bias=first.bias is not None,
        )
        with torch.no_grad():
            w = first.weight                        # (out, 3, k, k)
            new.weight.copy_(torch.cat([w, w], dim=1) * 0.5)
            if first.bias is not None:
                new.bias.copy_(first.bias)
        # Remplace le module dans l'arborescence.
        parent = backbone
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, new)

    def forward(self, mammo, us, clinical=None):
        x = torch.cat([mammo, us], dim=1)           # (B, 6, H, W)
        feats = self.encoder(x)
        if self.clinical_encoder is not None:
            feats = torch.cat([feats, self.clinical_encoder(clinical)], dim=1)
        return self.head(feats).squeeze(1)


# ---------------------------------------------------------------------------
# b. Concaténation tardive
# ---------------------------------------------------------------------------
class LateFusion(_FusionBase):
    """Embeddings concaténés avant le classifieur — l'approche de Chen et al.

    Chaque modalité garde son extracteur ; les représentations sont concaténées
    juste avant la tête. Plus expressif que LogitAveraging (la tête peut
    combiner les caractéristiques de façon non linéaire) mais toujours sans
    attention explicite.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.head = _head(2 * self.embed_dim + self.clinical_out, self.dropout)

    def forward(self, mammo, us, clinical=None):
        f_m, f_u, f_c = self._encode(mammo, us, clinical)
        feats = [f_m, f_u] + ([f_c] if f_c is not None else [])
        return self.head(torch.cat(feats, dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# c. Cross-attention
# ---------------------------------------------------------------------------
class CrossAttentionFusion(_FusionBase):
    """Attention multi-têtes entre modalités — l'approche CUF-MT de Rashedi.

    Chaque modalité devient un jeton ; l'attention met à jour chaque jeton en
    fonction des autres. C'est le mécanisme dont la piste B doit précisément
    évaluer l'utilité : sur paires virtuelles, il n'y a pas d'interaction
    réelle à capter, donc tout gain par rapport à LogitAveraging devra être
    expliqué (et vérifié par le test d'intégrité).
    """

    def __init__(self, n_heads: int = 4, n_layers: int = 1, **kw):
        super().__init__(**kw)
        d = self.embed_dim

        # Le clinique est projeté dans la même dimension pour devenir un jeton.
        self.clin_proj = (
            nn.Linear(self.clinical_out, d) if self.clinical_out else None
        )
        self.n_tokens = 3 if self.clin_proj is not None else 2
        self.token_type = nn.Parameter(torch.randn(self.n_tokens, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            dropout=self.dropout, batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = _head(self.n_tokens * d, self.dropout)

    def _tokens(self, mammo, us, clinical):
        f_m, f_u, f_c = self._encode(mammo, us, clinical)
        toks = [f_m, f_u]
        if self.clin_proj is not None:
            toks.append(self.clin_proj(f_c))
        x = torch.stack(toks, dim=1)                # (B, T, D)
        return x + self.token_type.unsqueeze(0)

    def forward(self, mammo, us, clinical=None):
        x = self.attn(self._tokens(mammo, us, clinical))
        return self.head(x.flatten(1)).squeeze(1)


# ---------------------------------------------------------------------------
# d. Cross-attention + AMW
# ---------------------------------------------------------------------------
class CrossAttentionAMW(CrossAttentionFusion):
    """Cross-attention + pondération adaptative des modalités (Adaptive
    Modality Weighting) — la stratégie gagnante chez Rashedi et al.

    Après l'attention, un petit réseau produit un poids PAR CAS pour chaque
    modalité : sur telle patiente le modèle peut s'appuyer surtout sur l'écho,
    sur telle autre surtout sur la mammo. La sortie est la somme pondérée des
    jetons, plutôt que leur concaténation brute.

    Intérêt spécifique à la piste B : même sans interaction réelle entre
    modalités, la pondération par cas peut capter la QUALITÉ de chaque image
    (flou, contraste, lésion visible ou non) — c'est l'hypothèse qui pourrait
    faire mieux que LogitAveraging. Les poids sont exposés par
    `last_modality_weights` pour l'analyse du rapport (Rashedi trouve l'écho
    dominante : à vérifier chez nous).
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.embed_dim
        self.gate = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(), nn.Linear(d // 4, 1))
        self.head = _head(d, self.dropout)          # somme pondérée → dim d
        self.last_modality_weights: "torch.Tensor | None" = None

    def forward(self, mammo, us, clinical=None):
        x = self.attn(self._tokens(mammo, us, clinical))     # (B, T, D)
        w = torch.softmax(self.gate(x).squeeze(-1), dim=1)   # (B, T)
        self.last_modality_weights = w.detach()
        fused = (w.unsqueeze(-1) * x).sum(dim=1)             # (B, D)
        return self.head(fused).squeeze(1)


# ---------------------------------------------------------------------------
# Registre — pour piloter les expériences par config YAML
# ---------------------------------------------------------------------------
FUSION_REGISTRY = {
    "logit_averaging": LogitAveraging,      # baseline OBLIGATOIRE
    "early": EarlyFusion,
    "late": LateFusion,
    "cross_attention": CrossAttentionFusion,
    "cross_attention_amw": CrossAttentionAMW,
}


def build_fusion(name: str, **kw):
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Stratégie '{name}' inconnue. Choix : {list(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](**kw)
