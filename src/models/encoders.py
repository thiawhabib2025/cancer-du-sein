"""
Extracteurs de caractéristiques — les 5 architectures candidates du §6.2.

Toutes exposent la même interface : image → vecteur d'embedding. C'est ce qui
permet de changer d'architecture par un simple paramètre de config, sans
toucher au code de fusion.

REMARQUES SUR LE CHOIX DES ARCHITECTURES
----------------------------------------
Le cahier des charges en liste cinq. Deux avertissements pratiques :

**EfficientNet-B7 et ViT-B/16 sont disproportionnés pour ce projet.** Ils
comptent des dizaines de millions de paramètres et ont été conçus pour
ImageNet (1,2 M d'images). Sur BUSI (780 images) ou MIAS (322 images), ils
sur-apprendront presque certainement. Attendez-vous à ce que ReXNet-200 et
Swin-Tiny, plus légers, l'emportent — ce qui rejoindrait la conclusion
d'Alswilem & Pacal sur le compromis performance/efficacité.

**EfficientNet-B7 est entraîné en résolution native 600×600.** L'utiliser en
224×224, comme le prévoit le pipeline, dégrade nettement ses performances : ce
n'est pas une comparaison équitable. Soit vous l'évaluez à sa résolution native
(coût GPU multiplié par ~7), soit vous documentez explicitement la limite.
La seconde option est raisonnable pour un M1 ; la première est plus rigoureuse.

Le tableau §6.2 du cahier des charges contient par ailleurs des comptes de
paramètres erronés — utilisez `print_model_comparison()` pour obtenir les
valeurs réelles et corriger le document.
"""

from __future__ import annotations

import warnings

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch, nn = None, object

__all__ = ["MODEL_REGISTRY", "ImageEncoder", "ClinicalEncoderMLP", "MonoModalClassifier"]


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "resnet18": {
        "timm_name": "resnet18",
        "native_size": 224,
        "gradcam_layer": "layer4",
        "ref": "Chen et al. (2025) [17]",
    },
    "efficientnet_b7": {
        "timm_name": "tf_efficientnet_b7",
        "native_size": 600,          # ← 224 dégrade fortement ce modèle
        "gradcam_layer": "conv_head",
        "ref": "Latha et al. (2024) [16]",
    },
    "rexnet_200": {
        "timm_name": "rexnet_200",
        "native_size": 224,
        "gradcam_layer": "features",
        "ref": "Alswilem & Pacal (2025) [15]",
    },
    "vit_b16": {
        "timm_name": "vit_base_patch16_224",
        "native_size": 224,
        "gradcam_layer": "blocks.11.norm1",   # Grad-CAM → préférer Attention Rollout
        "ref": "Sherine & Revathy (2025) [21]",
    },
    "swin_tiny": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "native_size": 224,
        "gradcam_layer": "layers.3.blocks.1.norm1",
        "ref": "Sherine & Revathy (2025) [21]",
    },
}


# ---------------------------------------------------------------------------
# Encodeur image
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """Backbone timm sans tête de classification → vecteur d'embedding.

    `num_classes=0` demande à timm de renvoyer les caractéristiques après
    pooling global, ce qui donne la même interface quelle que soit
    l'architecture (CNN ou transformer).
    """

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        image_size: int = 224,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        import timm

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Architecture '{name}' inconnue. Disponibles : {list(MODEL_REGISTRY)}"
            )
        spec = MODEL_REGISTRY[name]
        self.name = name
        self.spec = spec

        native = spec["native_size"]
        if image_size != native:
            warnings.warn(
                f"{name} : résolution native {native}, utilisée en {image_size}. "
                + (
                    "Dégradation attendue — à documenter dans le rapport."
                    if native > image_size
                    else "Vérifiez la compatibilité (ViT/Swin exigent une taille fixe)."
                )
            )

        kwargs = dict(pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        # Seuls les ViT acceptent img_size (interpolation des embeddings de
        # position). Swin exige la taille de sa fenêtre et refuse l'argument.
        if spec["timm_name"].startswith("vit_"):
            kwargs["img_size"] = image_size

        try:
            self.backbone = timm.create_model(spec["timm_name"], **kwargs)
        except RuntimeError as exc:
            # Les noms de poids timm évoluent (tags .ns_jft_in1k, etc.).
            # Repli sans pré-entraînement plutôt qu'un plantage en plein criblage.
            warnings.warn(f"{spec['timm_name']} : {exc}. Repli sans pré-entraînement.")
            kwargs["pretrained"] = False
            self.backbone = timm.create_model(spec["timm_name"], **kwargs)

        self.out_dim: int = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)

    def freeze(self, unfreeze_last_n: int = 0) -> None:
        """Gèle le backbone, en laissant éventuellement les N derniers blocs libres.

        Utile en phase de criblage : geler accélère beaucoup l'entraînement.
        Ne pas l'utiliser pour les résultats finaux — le fine-tuning complet
        est nettement meilleur en imagerie médicale, le domaine étant très
        éloigné d'ImageNet.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            children = list(self.backbone.children())
            for m in children[-unfreeze_last_n:]:
                for p in m.parameters():
                    p.requires_grad = True

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Encodeur clinique
# ---------------------------------------------------------------------------
class ClinicalEncoderMLP(nn.Module):
    """MLP sur les variables cliniques.

    Volontairement petit : 5 à 10 variables d'entrée ne justifient pas un
    réseau profond, et un MLP surdimensionné mémoriserait le bruit du
    générateur synthétique. Deux couches suffisent.
    """

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.3):
        super().__init__()
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Classifieur mono-modal — baseline de la Phase 2
# ---------------------------------------------------------------------------
class MonoModalClassifier(nn.Module):
    """Encodeur image (+ clinique optionnel) → logit unique.

    C'est le modèle des baselines de la Phase 2. La fusion ne sera jugée que
    par rapport à lui : si elle ne le bat pas, elle n'a aucun intérêt.

    Sortie : logit brut, pas de sigmoïde. La perte `BCEWithLogitsLoss` est
    numériquement plus stable, et le calibrage par température s'applique
    directement sur le logit.
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
        self.image_encoder = ImageEncoder(encoder_name, pretrained, image_size)
        dim = self.image_encoder.out_dim

        self.clinical_encoder = None
        if clinical_dim > 0:
            self.clinical_encoder = ClinicalEncoderMLP(clinical_dim, dropout=dropout)
            dim += self.clinical_encoder.out_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, image, clinical=None):
        feats = self.image_encoder(image)
        if self.clinical_encoder is not None:
            if clinical is None:
                raise ValueError("Le modèle attend des variables cliniques.")
            feats = torch.cat([feats, self.clinical_encoder(clinical)], dim=1)
        return self.head(feats).squeeze(1)


class ClinicalOnlyClassifier(nn.Module):
    """Baseline clinique seule — sert aussi de contrôle anti-fuite.

    Si ce modèle dépasse 0,65 d'AUC alors que les variables sont synthétiques,
    le générateur fuit (voir clinical_synth.py, RÈGLE N°2). Ce n'est pas
    qu'un baseline : c'est un instrument de diagnostic.
    """

    def __init__(self, clinical_dim: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = ClinicalEncoderMLP(clinical_dim, dropout=dropout)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, clinical):
        return self.head(self.encoder(clinical)).squeeze(1)


# ---------------------------------------------------------------------------
# Utilitaire — corriger le tableau §6.2 du cahier des charges
# ---------------------------------------------------------------------------
def print_model_comparison(image_size: int = 224) -> "pd.DataFrame":
    """Compte réel des paramètres et dimension d'embedding de chaque candidat.

    À exécuter une fois et à reporter dans le cahier des charges : le tableau
    §6.2 annonce 22,92 M pour ResNet-18 (valeur qui correspond en réalité au
    modèle bi-encodeur complet de Chen et al.) et confond ResNet-200 avec
    ReXNet-200.

    ⚠️ Deux précautions avant de recopier les chiffres dans le rapport :

    1. `num_classes=0` retire la couche de classification. Les comptes obtenus
       ici sont donc ceux de l'**extracteur seul**, légèrement inférieurs aux
       valeurs publiées qui incluent la tête ImageNet à 1 000 classes
       (~0,5 M de paramètres pour ResNet-18, ~2,6 M pour EfficientNet-B7).
       Précisez laquelle des deux conventions vous rapportez.
    2. Ordres de grandeur attendus, à CONFIRMER par exécution — ne les recopiez
       pas sans avoir lancé la fonction :
       ResNet-18 ≈ 11 M · ReXNet-200 ≈ 16 M · Swin-Tiny ≈ 28 M ·
       EfficientNet-B7 ≈ 64 M · ViT-B/16 ≈ 86 M.
       Aucun ne s'approche des 22,92 M attribués à ResNet-18 par le §6.2.
    """
    import pandas as pd
    import timm

    rows = []
    for name, spec in MODEL_REGISTRY.items():
        try:
            m = timm.create_model(spec["timm_name"], pretrained=False, num_classes=0)
            rows.append({
                "nom": name,
                "timm": spec["timm_name"],
                "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                "embed_dim": m.num_features,
                "taille_native": spec["native_size"],
                "compatible_224": spec["native_size"] == 224,
                "reference": spec["ref"],
            })
        except Exception as exc:
            rows.append({"nom": name, "timm": spec["timm_name"], "params_M": None,
                         "embed_dim": None, "erreur": str(exc)[:60]})
    return pd.DataFrame(rows)
