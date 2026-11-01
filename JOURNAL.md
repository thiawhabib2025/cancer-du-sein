# Journal de bord

> Une entrée par séance de travail. Noter **les décisions et les impasses**, pas le détail du code.
> Ce fichier fournira une grande partie du rapport final — tenez-le à jour, c'est du temps gagné en novembre.
>
> Format : `## AAAA-MM-JJ — Titre` · qui / quoi / pourquoi / ce qui bloque.

---

## 2026-08-07 — Audit du cahier des charges

**Cinq verrous identifiés** (détail dans `00_PLAN_PROJET.md`, Partie 1) :

1. 🔴 **Aucun dataset public n'apparie mammographie et échographie chez la même patiente.**
   Les six datasets prévus sont six populations distinctes. La cross-attention inter-modale
   n'a donc aucune correspondance réelle à apprendre.
2. 🔴 **La génération clinique synthétique conditionnée au label produirait une fuite de cible.**
3. 🟠 Risque de fuite au niveau patiente si le split se fait par image.
4. 🟠 INbreast et UDIAT ne sont pas librement téléchargeables (délai de réponse imprévisible).
5. 🟡 Périmètre ≈ 2× ce qu'une équipe de 3 peut livrer en un semestre.

**Décisions prises** (à valider en réunion de cadrage) :

- Projet restructuré en **3 pistes** : A (image + clinique réelle, socle solide),
  B (tri-modal sur appariement virtuel, risque assumé et mesuré), C (données appariées réelles, pari).
- **CMMD** ajouté comme source principale de données cliniques réelles (1 775 patientes avec âge).
- **BrEaST** et **BUS-UCLM** préparés en repli d'UDIAT et BUI.
- Variables cliniques stratifiées en 3 niveaux **C0 / C1 / C2**, jamais mélangés dans les résultats.
- Périmètre découpé en **Noyau / Extension / Bonus** avec jalons Go/No-Go.

**En attente :** validation des encadrants sur les décisions n°1 et n°2.

---

## 2026-08-08 — PREMIER ENTRAÎNEMENT RÉUSSI : AUC 0,938 sur BUSI

ResNet-18, BUSI pli 0, 441 train / 112 val, T4, **1,8 min**.

| Métrique | Valeur |
|---|---|
| **AUC-ROC** | **0,9384** — IC95 % [0,884 ; 0,980] |
| Sensibilité / Spécificité (Youden) | 0,865 / 0,947 |
| Spécificité à 90 % de sensibilité | 77,3 % |
| Spécificité à 95 % de sensibilité | 72,0 % |
| Spécificité à 98 % de sensibilité | **29,3 %** |
| Meilleure époque | 11 (early stopping à 17) |

**Jalon G2 côté échographie : ATTEINT** (cible ≥ 0,90).

**Le chiffre qui compte cliniquement, ce n'est pas l'AUC.** À 98 % de
sensibilité — l'exigence réelle du dépistage, ne pas manquer de cancer — la
spécificité s'effondre à 29 % : on flaguerait 71 % des lésions bénignes. C'est
la marge de progression réelle du projet, et c'est cette courbe qu'il faudra
montrer aux radiologues, pas l'AUC.

**BUG RÉEL ATTRAPÉ PAR LE CONTRÔLE VISUEL (cellule 5).** Plusieurs images
d'entraînement présentaient un bruit **coloré** (pixels rouges/verts/bleus).
Cause : `A.GaussNoise` tirait le bruit indépendamment par canal, alors que nos
images sont en niveaux de gris dupliqués sur 3 canaux. Le modèle apprenait à
ignorer un artefact qui n'existe dans aucune échographie réelle. Corrigé avec
`per_channel=False`.

→ **Aucun test automatique n'aurait vu ça.** La cellule « regarder 12 images »
n'est pas décorative : elle vient de payer. À refaire à chaque changement de
prétraitement.

**Sur-apprentissage net** : train loss 0,93 → 0,22 pendant que la val loss
plafonne vers 0,60 dès l'époque 6. Normal sur 441 images. L'early stopping fait
son travail. À surveiller avec des architectures plus grosses (EfficientNet-B7,
ViT-B/16 vont probablement s'effondrer ici).

**Confirmation de l'erreur du CdC §6.2** : ResNet-18 = **11,18 M paramètres**,
et non 22,92 M comme l'annonce le tableau. La valeur du CdC correspond au
modèle bi-encodeur complet de Chen et al. À corriger dans le document.

**Deux corrections techniques** : warnings AMP (`torch.cuda.amp` → `torch.amp`,
déprécié depuis torch 2.4) et `git: nogit` dans runs.csv — le projet n'est pas
un dépôt git, donc la traçabilité par commit ne fonctionne pas. Faire
`git init` + premier commit avant les expériences de la Phase 2, sinon on ne
pourra pas relier un résultat à une version du code.

---

## 2026-08-08 — LES 6 DATASETS INTÉGRÉS : 7 879 images / 4 278 patientes

Tous les datasets sont téléchargés et validés dans le pipeline. Manifeste final :

| Dataset | Images | Patientes | Malignes | Particularité gérée |
|---|---|---|---|---|
| CBIS-DDSM | 2 857 | 1 460 | 44,7 % | CSV Kaggle → JPEG via dicom_info.csv (le piège documenté, confirmé) |
| CMMD | 3 744 | 1 775 | 70,3 % | **Âge réel 100 %** ; diagnostic PAR SEIN (xlsx), côté lu dans l'en-tête DICOM |
| MIAS | 115 | 108 | 45,2 % | Paires gauche/droite regroupées |
| BUSI | 647 | 647 | 32,5 % | Limite patiente documentée |
| BrEaST | 252 | 252 | 38,9 % | Âge + BI-RADS réels ; "not available" → NaN |
| BUS-UCLM | 264 | 36 | 34,1 % | Préfixe fichier = patiente (~7 img/patiente) ; INFO.csv `;` |

Total : **6 716 mammographies + 1 163 échographies**, 3 955 lignes avec âge
réel (50 %), 2 637 masques. Splits 5 plis + hold-out : zéro fuite, taux de
malignes 0,54-0,57 partout. Un échantillon de chaque dataset chargé et vérifié
(y compris DICOM CMMD lu directement, 2294×1914).

**Corrections faites en intégrant le réel :**
1. CBIS-DDSM Kaggle : les CSV pointent des DICOM inexistants → résolution des
   SeriesUID via dicom_info.csv, filtrage 'full mammogram' vs 'ROI mask'.
   429 anomalies multiples agrégées au niveau image.
2. CMMD : le xlsx donne un diagnostic par (patiente, sein) — 1 458 DICOM de
   seins sains controlatéraux correctement exclus. Scan des 5 202 en-têtes
   DICOM par lots reprenables (montage lent) : scripts/cmmd_scan_batch.py.
3. Masques BUS-UCLM en RGB : IMREAD_GRAYSCALE perdait le rouge (malignes !)
   — 0,299×255≈76<127. Corrigé : lecture couleur + max canaux.
4. BreastDataset lit désormais les .dcm directement (LUT appliquées) ;
   conversion PNG recommandée avant entraînement intensif (~5× plus rapide).
5. Préfixe manifeste = nom réel du dossier (CBIS-DDSM ≠ clé cbis_ddsm).

**Attention pour l'entraînement :** CMMD est à 70 % de malignes (cohorte
biopsiée) contre 45 % pour CBIS — le déséquilibre inter-datasets est un
facteur de confusion potentiel : un modèle peut apprendre "ressemble à CMMD
⇒ malin". À surveiller dans l'analyse par dataset (Phase 2).

---

## 2026-08-08 — BUSI validé : 762 images réelles dans le pipeline

BUSI téléchargé (dossier `Dataset_BUSI_with_GT`, renommé en `data/raw/busi/`).

- **647 images / 210 malignes** — exactement les effectifs attendus
  (437 bénignes + 210 malignes, les 133 normales exclues par le parser).
- 647/647 masques de segmentation associés, binaires (0/255), lisibles.
  Ils serviront à la validation IoU du Grad-CAM en Phase 4.
- Splits combinés MIAS+BUSI : 762 images / 755 patientes, plis équilibrés
  (0,31-0,37 de malignes), zéro fuite.

**Bug attrapé par les vraies données (2e) :** `build()` préfixait
`image_path` par le nom du dataset mais pas `mask_path` — les masques
devenaient introuvables. Détecté parce que la vérification a tenté d'OUVRIR
un masque au lieu de se contenter de compter les chemins non vides. Corrigé.

Rappel de la limite BUSI (documentée dans le parser) : correspondance
image→patiente non publiée, chaque image traitée comme une patiente distincte.

---

## 2026-08-08 — PREMIÈRES DONNÉES RÉELLES : MIAS validé de bout en bout

MIAS téléchargé par Habib et passé dans le vrai pipeline. Résultats :

- **115 images anormales / 108 patientes / 52 malignes (45,2 %)** — cohérent
  avec la littérature (~115 anormales, ~51-54 malignes).
- Toutes les images trouvées, toutes lisibles, toutes en 1024×1024 uint8.
- Splits 5 plis + hold-out : **aucune fuite patiente**, taux de malignes
  équilibré entre plis (0,38-0,55).
- `BreastDataset._load_image` fonctionne sur les vrais .pgm.

**Bug réel attrapé par la confrontation aux vraies données :** Info.txt
contient ~330 entrées pour 322 images — certaines images portent plusieurs
lésions, ce que mes données factices ne simulaient pas. Le parser créait des
doublons d'image (détectés par le warning de `validate()`). Corrigé :
agrégation au niveau image, label le plus sévère retenu (8 lignes agrégées).

Leçon confirmée : les tests factices valident la logique, seules les vraies
données valident les hypothèses sur leur format.

**Placement des fichiers :** les datasets vont dans `data/raw/<nom>/` à la
racine du projet (PAS dans `src/data/raw/`). Pour MIAS : les .pgm et Info.txt
directement dans `data/raw/mias/`.

---

## 2026-08-08 — DÉCISIONS D'ÉQUIPE : piste B + clinique C0 uniquement

**Décision n°1 → Piste B.** L'architecture principale du projet est la fusion
tri-modale (mammo + écho + clinique) sur **appariement virtuel**, conformément
à l'ambition du cahier des charges. Conséquences immédiates :

- Le baseline **moyenne des logits** devient obligatoire dans toute expérience
  de fusion — c'est lui qui dira si la cross-attention apporte quelque chose.
- Le protocole complet (5 tirages d'appariement, test d'intégrité par
  appariement aléatoire, DeLong systématique) est documenté dans
  `docs/protocole_piste_B.md`.
- L'appariement virtuel est nommé et assumé dans le rapport — jamais présenté
  comme une fusion sur données réelles.
- Note : la piste A (image + clinique réelle) reste couverte de fait par les
  baselines mono-modaux avec variables cliniques — aucun travail supplémentaire.

**Décision n°2 → C0 uniquement.** Seules les variables cliniques **réelles**
entrent dans le modèle : âge (CMMD, BrEaST), densité BI-RADS (CBIS-DDSM,
BrEaST), type de lésion (CBIS-DDSM, CMMD). Conséquences :

- Antécédents familiaux, BRCA et IMC **sortent du périmètre** — le CdC §5.4
  est amendé. À signaler aux encadrants.
- `clinical_synth.py` et son garde-fou restent dans le dépôt mais ne sont
  **plus sur le chemin critique** : le risque de fuite clinique disparaît
  entièrement puisqu'on n'invente plus rien.
- Le MLP clinique s'entraîne sur 3 variables réelles. Son AUC attendue est
  modeste (~0,55-0,62) : c'est normal et c'est même la preuve que les données
  sont honnêtes.

---

## 2026-08-08 — Socle Phase 2 : modèles, métriques, entraînement

Ajout de `src/models/encoders.py`, `src/eval/{metrics,delong}.py`,
`src/train/trainer.py`. **41 tests au vert.**

**Deux décisions méthodologiques prises ici.**

1. **Le seuil 0,5 est abandonné au profit d'un point de fonctionnement clinique.**
   La quasi-totalité des articles rapporte sensibilité et spécificité au seuil
   0,5, hérité du softmax. En dépistage on fixe d'abord la sensibilité qu'on
   s'impose, puis on lit la spécificité. `specificity_at_sensitivity()` répond
   à la vraie question : « combien de biopsies inutiles évite-t-on à 95 % de
   sensibilité ? ». C'est la métrique à présenter aux radiologues.

2. **Les intervalles de confiance sont bootstrappés par patiente, pas par image.**
   Vérifié expérimentalement sur données simulées (167 patientes × 3 vues
   corrélées) : un bootstrap par image ne produit que **59 % de la largeur
   réelle** de l'intervalle. Il sous-estime l'incertitude d'un facteur 1,7.
   Même famille d'erreur que le découpage par image, en plus discret.

**DeLong vérifié** contre trois références indépendantes : AUC identique à
sklearn (écart 2e-16), variance cohérente avec l'approximation de
Hanley-McNeil (ratio 0,87), et p = 1 exactement pour un modèle comparé à
lui-même. Antisymétrie confirmée.

**Observation utile pour plus tard :** sur données simulées, un écart d'AUC de
0,053 sur 500 échantillons donne p = 0,072 — **non significatif**. Cela
préfigure ce qui nous attend en Phase 3 : un gain apparent de la fusion de
l'ordre de 0,02–0,05 pourra très bien ne rien signifier. D'où le jalon G3, qui
exige la significativité et pas seulement l'écart.

**À faire :** `print_model_comparison()` n'a pas pu être exécuté (torch non
installable dans l'environnement de développement). À lancer sur Colab pour
obtenir les vrais comptes de paramètres et corriger le tableau §6.2 du CdC.

---

## 2026-08-08 — Abandon d'INbreast, UDIAT et BUI

**Décision.** Les trois datasets à accès restreint sont retirés du projet.

**Motif.** INbreast et UDIAT nécessitent une demande aux auteurs avec un délai de
réponse imprévisible ; BUI n'a jamais eu de source vérifiable. Ils constituaient
le principal risque de planning, et deux des trois configurations de
généralisation du CdC §5.5 en dépendaient.

**Remplacements** (tous librement téléchargeables) :

| Retiré | Remplacé par | Effet |
|---|---|---|
| INbreast (115 cas) | CMMD (1 775 patientes) | ×15 en effectif + âge réel |
| UDIAT (163 images) | BrEaST (256 patientes) | Âge, BI-RADS, taille de lésion réels |
| BUI | BUS-UCLM | Source vérifiable (Sci Data 2025) |

**Conséquences.**

- Le protocole de généralisation est reconstruit et **étendu à l'échographie**
  (3 configs mammo + 3 configs écho), ce que le CdC ne prévoyait pas.
- Perte apparente : INbreast était annoncé comme seule source de densité BI-RADS
  réelle. En fait CBIS-DDSM contient déjà un champ `breast density` (1-4) —
  la densité réelle reste disponible.
- **Plus aucune tâche du projet n'attend une réponse extérieure**, hors piste C
  (données appariées réelles), qui reste un bonus et non un prérequis.

**À faire :** vérifier les noms de colonnes réels de BrEaST et BUS-UCLM au
téléchargement — les parsers ont été écrits sans avoir vu les fichiers et sont
volontairement tolérants, mais doivent être confirmés.

---

## 2026-08-07 — Socle technique

Dépôt initialisé. Priorité donnée aux deux garde-fous méthodologiques avant tout code de modèle :

- `src/data/splits.py` — découpage par `patient_id` via `StratifiedGroupKFold`
- `src/data/clinical_synth.py` — génération à partir de prévalences et odds ratios publiés,
  avec auto-test du plafond AUC ≤ 0,65

**À faire :** exécuter les demandes d'accès INbreast/UDIAT et lancer les démarches hôpital (piste C).

---

<!-- Nouvelles entrées au-dessus de cette ligne -->
