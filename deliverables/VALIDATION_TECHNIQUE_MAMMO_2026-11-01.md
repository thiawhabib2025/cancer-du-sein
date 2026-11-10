# Validation technique — modèle mammographique CMMD

**Date :** 1 novembre 2026  
**Périmètre :** chargement sûr, correspondance d'architecture, prétraitement et inférence de contrôle sur un exemple du train.  
**Statut :** validation technique réussie ; aucune évaluation de performance ni validation clinique.

## Résumé

Le checkpoint `mammo_seule_best.pt` a été chargé avec `torch.load(..., weights_only=True, map_location="cpu")`. Ses 122 tenseurs ont été chargés sans clé manquante ni inattendue dans le modèle actuel `MonoModalClassifier` configuré en ResNet-18, avec `strict=True`.

Une image CMMD du split d'entraînement a suivi le pipeline d'évaluation déterministe du notebook. Le modèle a produit un logit fini et une probabilité maligne finie, avec les formes attendues. Cette image unique vérifie le fonctionnement de la chaîne technique ; elle ne permet pas de mesurer la généralisation ni les performances du modèle.

## Checkpoint et reconstruction

- **Fichier :** `checkpoints/mammo_seule_best.pt` (134 265 633 octets)
- **SHA-256 :** `6f55646587ca5de6a4e31a32609d7299302d15cfab00684d960ca8fe70e6c344`
- **Run enregistré :** `mammo_seule`
- **Époque stockée :** 12
- **Modèle :** `MonoModalClassifier(encoder_name="resnet18", image_size=224, clinical_dim=0, dropout=0.3)`
- **Encodeur :** `timm` ResNet-18, `num_classes=0`, embedding de 512 dimensions
- **Tête :** `Dropout(0.3)` puis `Linear(512, 1)` ; sortie logit brute, sans sigmoïde dans le modèle
- **Nombre de paramètres reconstruit :** 11 177 025
- **Chargement :** strict réussi ; le `state_dict` contient 122 tenseurs

Pour le chargement, l'encodeur a été créé avec `pretrained=False` : les poids ImageNet externes ne sont pas téléchargés, et le checkpoint fournit ensuite l'ensemble des poids via la correspondance stricte.

### Paramètres d'entraînement présents dans le checkpoint

`epochs=15`, `lr=0.0003`, `weight_decay=0.0001`, `batch_size=32`, `patience=5`, `amp=true`, `grad_clip=1.0`, `monitor="auc_roc"`, `seed=42`, `warmup_epochs=2`, `pos_weight=null`.

Le champ `score` du checkpoint vaut `0.7528`. Il s'agit d'une valeur enregistrée avec les poids, pas d'une métrique recalculée pendant cette validation. Elle n'est donc pas présentée ici comme un nouveau résultat vérifié.

## Environnement utilisé

Environnement virtuel distinct du dépôt, situé sous `%LOCALAPPDATA%\\Temp\\mammo-validation-venv` :

| Composant | Version |
|---|---:|
| Python | 3.13.9 |
| PyTorch | 2.7.1+cpu |
| torchvision | 0.22.1+cpu |
| timm | 1.0.15 |
| Albumentations | 2.0.8 |
| NumPy | 2.2.6 |
| pandas | 3.1.0 |
| OpenCV | 5.0.0 |
| pydicom | 3.0.2 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.9.1 |

Le test a été exécuté sur CPU ; CUDA n'était pas disponible dans cet environnement. Aucune dépendance du dépôt n'a été modifiée.

## Prétraitement retracé

Le protocole du notebook [`04_resultats_presentation.ipynb`](../notebooks/04_resultats_presentation.ipynb) et le code du projet indiquent :

1. Lecture DICOM et application de la VOI LUT ; inversion de `MONOCHROME1` si nécessaire ; mise à l'échelle de l'image vers `uint16`.
2. Division par 257 vers `uint8`, redimensionnement à 256×256 avec `INTER_AREA`, puis stockage/relecture PNG en niveaux de gris.
3. Conversion en RGB.
4. Pour validation et inférence : redimensionnement à 224×224, normalisation ImageNet (moyennes `[0.485, 0.456, 0.406]`, écarts-types `[0.229, 0.224, 0.225]`) et conversion en tenseur.

Pendant l'entraînement, le notebook ajoute à la taille 224×224 : flip horizontal (p=0,5), rotation ±15° (p=0,7), variation de luminosité/contraste ±0,15 (p=0,5) et bruit gaussien (p=0,2, partagé entre les canaux). Aucun flip vertical n'est appliqué.

Le checkpoint ne stocke pas de description du prétraitement. Le pipeline ci-dessus est celui retrouvé dans le notebook et rejoué pour ce test ; son identité historique exacte avec le pipeline ayant produit les poids ne peut pas être prouvée uniquement à partir du checkpoint.

## Test d'inférence contrôlé

- **Exemple :** ligne 556 de `data/manifest.csv`, CMMD, fold 0, label réel bénin (`label=0`). Le chemin de l'image n'est pas recopié dans ce rapport ; son empreinte SHA-256 est `b376f3ab07d89be485201db6087d6dc77627346cb0d779c70943aadbf4b96374`.
- **Image décodée :** 2294×1914 ; conversion DICOM réussie.
- **Tenseur d'entrée :** `[1, 3, 224, 224]`, toutes les valeurs finies.
- **Sortie du réseau :** un logit fini, `-1.21875894`.
- **Sigmoïde, classe maligne :** `0.22815493`, valeur finie dans `[0, 1]`.
- **Classe au seuil indicatif 0,5 :** bénin ; identique au label de cet exemple.
- **Split CMMD vérifié dans le manifeste :** train 2 570 images / 1 216 patientes ; validation 642 / 304 ; hold-out 532 / 255. Les identifiants patientes sont disjoints entre ces trois ensembles.

L'exemple appartient au train (fold 0). Le hold-out (fold -1) n'a été ni utilisé ni consulté pour ajuster le modèle. L'accord entre la prédiction et le label sur une seule image d'entraînement n'est pas une mesure de performance.

## Limites de provenance et erreurs observées

- La désérialisation sûre, la reconstruction de l'architecture, le chargement strict et l'inférence ont réussi.
- Le décodeur DICOM émet un avertissement de dépréciation : `pydicom.pixel_data_handlers` sera retiré dans pydicom 4 ; le code conseille désormais `pydicom.pixels.apply_voi_lut`. Cela n'a pas empêché le décodage et aucune modification du code source n'a été faite dans ce travail.
- Le checkpoint contient `git="nogit"` et sa configuration ne comporte pas de fingerprint de protocole/split. Le notebook actuel reconstruit la séparation par patiente et le prétraitement, mais la provenance complète des poids et l'identité exacte de leur version de code restent non démontrées.
- Le test ne remplace ni une évaluation complète sur la validation, ni l'évaluation finale sur le hold-out indépendant, ni une validation clinique.

## Modifications

Seul ce rapport a été créé. `demo_app.py`, les métriques du mémoire, le checkpoint et les fichiers source n'ont pas été modifiés.
