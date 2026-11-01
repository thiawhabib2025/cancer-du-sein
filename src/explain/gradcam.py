"""
Grad-CAM — cartes de chaleur des régions influençant la décision.

CHOIX D'IMPLÉMENTATION
----------------------
Implémentation autonome plutôt que la bibliothèque `pytorch-grad-cam` : le
point délicat est la sélection de la couche cible, qui diffère d'une
architecture `timm` à l'autre. Une détection automatique (dernière convolution
du backbone) est ici plus robuste qu'une table de correspondance à maintenir.

Principe : on capture les activations $A^k$ de la dernière couche
convolutive et les gradients du logit par rapport à ces activations. Le poids
de chaque canal est la moyenne spatiale de son gradient, et la carte est la
somme pondérée des canaux, passée par une ReLU :

    L = ReLU( Σ_k α_k A^k )   avec   α_k = moyenne_spatiale( ∂y/∂A^k )

⚠️ **Limite sur les transformers.** Grad-CAM suppose une carte de
caractéristiques spatiale (B, C, H, W). Les ViT produisent des jetons
(B, N, C) : la carte obtenue après remise en forme est approximative.
Pour ViT/Swin, l'*Attention Rollout* est plus fidèle. Si l'architecture
retenue est un transformer, signalez cette limite dans le rapport.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch, nn, F = None, None, None

__all__ = ["GradCAM", "find_target_layer", "overlay_cam"]


# ---------------------------------------------------------------------------
def find_target_layer(module) -> "nn.Module":
    """Dernière couche convolutive du module — cible par défaut de Grad-CAM.

    Plus robuste qu'un chemin codé en dur : fonctionne pour ResNet, ReXNet,
    EfficientNet et toute architecture convolutive de `timm`, sans table de
    correspondance à maintenir.
    """
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return convs[-1]

    # Architecture purement attentionnelle : dernière normalisation de couche.
    norms = [m for m in module.modules() if isinstance(m, nn.LayerNorm)]
    if norms:
        warnings.warn(
            "Aucune convolution trouvée (architecture transformer) : Grad-CAM "
            "sera approximatif. Préférer l'Attention Rollout, et le signaler "
            "dans le rapport."
        )
        return norms[-1]

    raise ValueError("Aucune couche cible exploitable dans ce module.")


# ---------------------------------------------------------------------------
class GradCAM:
    """Générateur de cartes Grad-CAM.

    Exemple
    -------
    >>> cam = GradCAM(model, model.image_encoder.backbone)
    >>> heatmap = cam(images)          # (B, H, W), valeurs dans [0, 1]
    >>> cam.remove()                   # libère les hooks
    """

    def __init__(self, model, module=None, target_layer=None):
        self.model = model
        self.layer = target_layer or find_target_layer(module or model)
        self._acts = None
        self._grads = None

        self._h1 = self.layer.register_forward_hook(self._save_activation)
        # register_full_backward_hook : la variante non-« full » est dépréciée
        # et silencieusement incorrecte sur certains modules.
        self._h2 = self.layer.register_full_backward_hook(self._save_gradient)

    # -- hooks ------------------------------------------------------------
    def _save_activation(self, _module, _inp, out):
        self._acts = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0].detach()

    # -- utilitaires ------------------------------------------------------
    @staticmethod
    def _to_spatial(t: "torch.Tensor") -> "torch.Tensor":
        """Ramène des jetons (B, N, C) en carte (B, C, H, W) si nécessaire."""
        if t.dim() == 4:
            return t
        if t.dim() == 3:
            b, n, c = t.shape
            side = int(round(n ** 0.5))
            if side * side != n:          # jeton de classe en tête (ViT)
                n -= 1
                t = t[:, 1:, :]
                side = int(round(n ** 0.5))
            return t.transpose(1, 2).reshape(b, c, side, side)
        raise ValueError(f"Forme d'activation inattendue : {tuple(t.shape)}")

    # -- appel ------------------------------------------------------------
    def __call__(self, inputs, output_size=None, forward_fn=None) -> np.ndarray:
        """Retourne les cartes normalisées, de forme (B, H, W) dans [0, 1].

        Parameters
        ----------
        forward_fn
            Passe avant personnalisée, pour les modèles de fusion dont la
            signature n'est pas ``model(x)``. Doit renvoyer les logits.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = forward_fn() if forward_fn is not None else self.model(inputs)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        # Rétropropagation de la somme des logits : chaque échantillon du lot
        # reçoit le gradient de son propre logit, les exemples étant indépendants.
        logits.sum().backward(retain_graph=False)

        if self._acts is None or self._grads is None:
            raise RuntimeError("Aucune activation capturée — couche cible non traversée.")

        acts = self._to_spatial(self._acts)
        grads = self._to_spatial(self._grads)

        weights = grads.mean(dim=(2, 3), keepdim=True)      # α_k
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))

        size = output_size or (inputs.shape[-2], inputs.shape[-1])
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Normalisation par échantillon, dans [0, 1]
        b = cam.shape[0]
        flat = cam.view(b, -1)
        lo = flat.min(dim=1, keepdim=True)[0]
        hi = flat.max(dim=1, keepdim=True)[0]
        cam = ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)

        return cam.cpu().numpy()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# ---------------------------------------------------------------------------
def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpose une carte de chaleur (jet) à une image, pour les figures.

    `image` : (H, W) ou (H, W, 3), uint8 ou float dans [0, 1].
    `cam`   : (H, W) dans [0, 1].
    """
    import cv2

    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if cam.shape != img.shape[:2]:
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    out = (1 - alpha) * img + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
