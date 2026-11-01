# Fusion multi-modale pour la détection précoce du cancer du sein

Projet de recherche comparant des modèles de mammographie, d'échographie et
des stratégies de fusion, avec des données cliniques observées et des outils
d'explicabilité (Grad-CAM + SHAP). Ce dépôt est un prototype de recherche,
pas un dispositif médical.

**EPT — M1 ILIA** · Fallou DIOUCK · Habib THIAW · Abibatou SYLLA

---

## ⚠️ À lire avant de coder

Deux règles méthodologiques conditionnent la validité de **tous** les résultats du projet.
Elles sont chacune protégées par un test automatique qui doit rester au vert.

### 1. Le découpage se fait par patiente, jamais par image

CBIS-DDSM contient plusieurs images par patiente (vues CC et MLO, image pleine, ROI, masque).
Un découpage au niveau image place la vue CC d'une patiente en entraînement et sa vue MLO en test :
le modèle mémorise la patiente au lieu d'apprendre la lésion.

➜ Utiliser exclusivement `src.data.splits.make_folds()`, qui repose sur `StratifiedGroupKFold(groups=patient_id)`.
➜ Garde-fou : `tests/test_no_patient_leakage.py`

### 2. Les données cliniques synthétiques ne doivent pas encoder le label

Générer l'âge, les antécédents ou BRCA conditionnellement au label crée une **fuite de cible** :
le modèle apprend à inverser le générateur, pas à détecter un cancer. Les AUC deviennent fictives.

➜ Plafond de contrôle : un classifieur entraîné sur les seules variables cliniques synthétiques
doit rester **sous 0,65 d'AUC** — l'ordre de grandeur des modèles de risque réels (type Gail, ~0,58–0,62).
➜ Garde-fou : `tests/test_clinical_no_leakage.py`

> Les variables synthétiques (niveau **C2**) ne figurent **jamais** dans les résultats principaux.
> Elles servent uniquement d'étude d'ablation clairement étiquetée. Voir `00_PLAN_PROJET.md`, Décision n°2.

---

## Périmètre retenu et limites

Le protocole opérationnel est recentré sur **CMMD** (mammographie) et **BUSI**
(échographie), conformément au notebook de résultats et aux données présentes
dans `data/raw/`. Les autres parsers sont conservés, mais ces jeux ne sont pas
activés dans `configs/data.yaml`.

- Aucune paire clinique réelle ne relie CMMD et BUSI. La fusion associe
  aléatoirement des patientes différentes, sans consulter les labels pour
  former les paires. La cible de fusion est le diagnostic mammographique ;
  les résultats ne permettent pas de conclure à un bénéfice clinique chez une
  patiente réellement examinée dans les deux modalités.
- CMMD fournit l'âge et le type radiologique de l'anomalie. BUSI ne fournit pas
  d'identifiant patient exploitable ; le parser emploie donc un identifiant par
  image. Le découpage BUSI ne garantit pas l'absence de vues corrélées entre
  plis. Cette limite doit accompagner tout résultat BUSI.
- Seules les variables observées sont admissibles. Les variables synthétiques
  C2 restent hors des résultats principaux.

Le découpage est groupé par `patient_id` via
[`src/data/splits.py`](./src/data/splits.py). Cela évite les recouvrements
pour les identifiants disponibles ; cela ne répare pas l'absence d'identifiants
réels BUSI.

## Expériences GPU

Le notebook [`notebooks/04_resultats_presentation.ipynb`](./notebooks/04_resultats_presentation.ipynb)
décrit le plan de 29 expériences : 3 baselines, 5 stratégies de fusion × 5
graines d'appariement et un contrôle par permutation des échographies du
hold-out, sans réentraînement. Il exige un runtime CUDA compatible PyTorch et
timm. Les checkpoints par époque, historiques, prédictions et métriques sont
sauvegardés séparément ; les fingerprints empêchent de reprendre un run avec
un protocole différent. Les performances ne sont pas incluses dans ce dépôt
tant qu'elles n'ont pas été mesurées et vérifiées sur le runtime.

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Le test structurel des modèles de fusion est ignoré si PyTorch n'est pas
installé. Un test vert sans PyTorch ne valide donc pas les passes avant/arrière
des réseaux ; exécuter les tests avec les dépendances de
[`requirements.txt`](./requirements.txt) avant tout entraînement.

## Reconstituer le manifeste CMMD + BUSI

Depuis la racine `breast-multimodal-fusion/`, avec les deux jeux placés dans
`data/raw/cmmd/` et `data/raw/busi/` :

```powershell
python -m src.data.build_manifest --config configs/data.yaml --out data/manifest.csv
python -m src.data.splits --manifest data/manifest.csv --n-splits 5
```

Le premier script lit les en-têtes DICOM et le fichier clinique CMMD ; il peut
prendre plusieurs minutes. Le second réserve un hold-out patient (fold `-1`)
et assigne cinq plis au reste. Ne lancez pas `clinical_synth.py` pour les
résultats principaux.

## Le manifeste

Fichier pivot. Une ligne = une image. `image_path` et `mask_path` sont relatifs
à `data/raw/`.

| Colonne | Type | Description |
|---|---|---|
| `patient_id` | str | **Préfixé par le dataset** (`cbis_P00123`) pour éviter les collisions inter-datasets |
| `image_path` | str | Chemin relatif depuis `data/raw/` |
| `dataset` | str | Dataset source (`cmmd` ou `busi` dans l'expérience retenue) |
| `modality` | str | `mammography` ou `ultrasound` |
| `view` | str | `CC`, `MLO`, `US`, `unknown` |
| `laterality` | str | `L`, `R`, `unknown` |
| `label` | int | 0 = bénin, 1 = malin |
| `lesion_type` | str | `mass`, `calcification`, `both`, `unknown` |
| `age` | float | **Réel** si disponible (C0), sinon `NaN` |
| `density` | str | BI-RADS A/B/C/D — réel si disponible |
| `mask_path` | str | Masque de lésion, sinon vide. Sert à valider le Grad-CAM par IoU. |
| `clinical_level` | str | `C0` (réel), `C1` (dérivé), `C2` (synthétique) |

## Datasets retenus

| Dataset | Modalité | Effectif | Clinique réelle |
|---|---|---|---|
| CMMD | Mammographie | 3 744 images / 1 775 patientes | Âge, type d'anomalie |
| BUSI | Échographie | 647 images anormales | Aucune variable clinique |

Les effectifs sont ceux du manifeste reconstruit localement ; les labels sont
déséquilibrés entre datasets (environ 70 % de positifs dans CMMD contre 32 %
dans BUSI). Un modèle peut apprendre un raccourci de provenance. Il faut donc
rapporter les résultats par dataset et éviter d'interpréter la fusion virtuelle
comme un gain médical.

Les notebooks sont des prototypes **non exécutés dans cet espace de travail** :
ils ne contiennent pas de prédictions ni de résultats finaux vérifiables.
Quatre checkpoints de référence (`busi_resnet18_fold0`, `echo_seule`,
`mammo_seule`, `clinique_seule`) ont été récupérés depuis Drive et leur
intégrité ZIP vérifiée, mais leur
chargement strict et l'inférence ne sont pas encore validés. Ils ne contiennent
pas d'empreinte du protocole et ne prouvent donc pas que les expériences
correspondent à la version actuelle du notebook.
Le journal rapporte un ancien essai BUSI (AUC 0,9384), mais ses artefacts
d'entraînement sont absents ; cette valeur est une trace de journal, pas un
résultat reproduit ici. Aucun résultat de fusion ne doit être présenté avant
une exécution reproductible.

Le rapport d'état et le protocole de continuation sont dans
[`RAPPORT_ETAT_PROJET.md`](../RAPPORT_ETAT_PROJET.md). Le guide Colab et le
guide de soutenance ont été actualisés pour ne pas reprendre les résultats
non vérifiés. Une présentation factuelle, sans chiffres expérimentaux
inventés, est disponible dans
[`Soutenance_Fusion_MultiModale_actualisee.pptx`](../Soutenance_Fusion_MultiModale_actualisee.pptx).
Des copies actualisées des rapports Word sont disponibles ici :
[`Rapport_orientation_projet_IA_mammographie_echographie_actualise.docx`](../Rapport_orientation_projet_IA_mammographie_echographie_actualise.docx)
et [`rapport_orientationVersionClaude_actualise.docx`](../rapport_orientationVersionClaude_actualise.docx).
Les versions originales sont conservées comme archives, mais contiennent des
résultats non vérifiables.

## Conventions

1. **Zéro constante en dur.** Chemins et hyperparamètres vivent dans `configs/`.
2. **Chaque entraînement écrit une ligne dans `results/`** avec le hash du commit, le nom de la config et le seed. En l'absence d'un dépôt Git propre, le hash est `nogit`.
3. **Aucune fusion sur `main` sans relecture** par un autre membre de l'équipe.
4. `data/raw/` et `data/processed/` ne sont **jamais** commités.

## Répartition

| Membre | Lot |
|---|---|
| Fallou DIOUCK | Données & infrastructure — `src/data/` |
| Habib THIAW | Modélisation & fusion — `src/models/`, `src/train/` |
| Abibatou SYLLA | Explicabilité — `src/explain/` |

**Démonstration :** `demo_app.py` propose trois onglets — aperçu des images,
méthodologie/limites et état technique. Elle accepte séparément une
mammographie et une échographie anonymisées (PNG, JPEG ou TIFF), vérifie le
format, le poids et les dimensions avant l'aperçu, et permet d'effacer les
fichiers de la session. La limite d'import est de 20 Mo et cette interface
n'envoie aucune image à un modèle.

Quatre checkpoints sont présents localement et leur intégrité d'archive a été
contrôlée, mais aucun n'a été chargé strictement ni validé par une inférence
contrôlée. PyTorch et timm sont absents de l'environnement local vérifié ;
l'inférence est donc désactivée. L'application n'affiche aucun score,
diagnostic ou Grad-CAM simulé. Elle demeure un prototype de recherche non
validé cliniquement, pas un dispositif médical.

Lancer l'interface depuis la racine du dépôt :

```powershell
python -m pip install streamlit Pillow
streamlit run demo_app.py
```

Tester le parcours de l'interface, le contrôle d'images et les garde-fous :

```powershell
python -m pytest tests/test_demo_readiness.py -q
```

Le rapport et la présentation de soutenance distincts sont dans
[`deliverables/`](./deliverables/) :

- [`MEMOIRE_SOUTENANCE_VERSION_2026-11-01.md`](./deliverables/MEMOIRE_SOUTENANCE_VERSION_2026-11-01.md)
- [`MEMOIRE_SOUTENANCE_VERSION_2026-11-01.docx`](./deliverables/MEMOIRE_SOUTENANCE_VERSION_2026-11-01.docx)
- [`SOUTENANCE_VERSION_2026-11-01.pptx`](./deliverables/SOUTENANCE_VERSION_2026-11-01.pptx)
- [`QUESTIONS_SOUTENANCE_2026-11-01.md`](./deliverables/QUESTIONS_SOUTENANCE_2026-11-01.md)

Les chiffres historiques cités sont explicitement exploratoires ; ils ne
remplacent pas une évaluation indépendante sur le hold-out. Les résultats
finaux ne doivent être ajoutés qu'après validation des checkpoints,
prédictions, identifiants patients et métriques.

Le rapport Word peut être régénéré depuis sa source Markdown :

```powershell
python -m pip install -r requirements-docs.txt
python scripts/build_soutenance_report.py
```

## Avertissement

Outil de recherche à visée d'aide à la décision. **Ne se substitue en aucun cas
à un diagnostic médical posé par un professionnel de santé.**
