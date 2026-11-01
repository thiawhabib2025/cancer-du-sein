# Guide de téléchargement des datasets

**Règle de placement :** chaque dataset va dans `data/raw/<nom>/` à la racine
du projet — pas dans `src/`. Le nom du dossier doit correspondre exactement à
la clé de `configs/data.yaml`.

```
breast-multimodal-fusion/
└── data/raw/
    ├── mias/            ✅ fait (322 .pgm + Info.txt directement dedans)
    ├── cbis_ddsm/
    ├── busi/
    ├── cmmd/
    ├── breast_poland/
    └── bus_uclm/
```

Après chaque téléchargement : `python -m src.data.build_manifest` et vérifier
que la ligne du dataset apparaît avec des effectifs plausibles.

---

## Par ordre de priorité

### 1. BUSI — échographie (~200 Mo, le plus simple)

- **Kaggle :** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- Source d'origine (Al-Dhabyani et al.) : https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
- **Placement :** les trois dossiers `benign/`, `malignant/`, `normal/`
  directement dans `data/raw/busi/`.
- Attendu : 780 images PNG + masques `*_mask.png`. Le parser exclut `normal`.

### 2. CBIS-DDSM — mammographie (~6 Go en version JPEG Kaggle, ~160 Go en DICOM TCIA)

- **Kaggle (recommandé pour démarrer) :**
  https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- Source officielle TCIA (DICOM complet) :
  https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- **Placement :** le contenu dans `data/raw/cbis_ddsm/`, en particulier les
  4 CSV (`mass_case_description_train_set.csv`, etc.) à la racine du dossier.
- ⚠️ **Piège connu de la version Kaggle :** les chemins d'images dans les CSV
  ne correspondent pas toujours à l'arborescence réelle des dossiers. Si
  build_manifest signale des chemins introuvables, c'est ce problème — me le
  signaler, le parser devra reconstruire la correspondance via les `SeriesUID`.

### 3. CMMD — mammographie + ÂGE RÉEL (~20 Go, le plus précieux)

- **TCIA :** https://www.cancerimagingarchive.net/collection/cmmd/
- Nécessite l'outil **NBIA Data Retriever** (téléchargement guidé sur la page).
- Ne pas oublier le fichier clinique **`CMMD_clinicaldata_revision.xlsx`**
  (lien « Clinical data » sur la même page) — c'est lui qui contient l'âge.
- **Placement :** DICOM dans `data/raw/cmmd/`, le xlsx à la racine du dossier.

### 4. BrEaST — échographie + âge/BI-RADS réels (~1 Go)

- **TCIA :** https://www.cancerimagingarchive.net/dataset/breast-lesions-usg/
- Publication : Pawłowska et al., Scientific Data 2024.
- **Placement :** images + masques + le fichier clinique
  (`BrEaST-Lesions-USG-clinical-data.xlsx` ou similaire) dans
  `data/raw/breast_poland/`.
- ⚠️ Le parser devine les noms de colonnes — au premier lancement de
  build_manifest, vérifier l'avertissement éventuel et me signaler les
  colonnes réelles.

### 5. BUS-UCLM — échographie (~600 Mo)

- **LIEN DIRECT (Mendeley Data, bouton « Download All ») :**
  https://data.mendeley.com/datasets/7fvgj4jsp7/3
- Publication : Vallez et al., Scientific Data 2025.
- **Placement :** contenu du zip dans `data/raw/bus_uclm/`.
- **Contenu vérifié sur la fiche Mendeley :** 683 images de **38 patientes
  seulement** (174 bénignes, 90 malignes, 419 normales), Siemens ACUSON S2000,
  2022-2023.
- ⚠️ **~18 images par patiente** : le regroupement par identifiant patiente est
  indispensable ici plus que partout ailleurs. À vérifier dans les noms de
  fichiers/métadonnées au téléchargement.
- ⚠️ **Masques RGB** (vert = bénin, rouge = malin, noir = fond) — différent des
  masques binaires de BUSI. Le parser devra convertir.
- **Rôle dans le projet :** trop petit pour entraîner (264 images utiles) →
  jeu de **test externe** des configs de généralisation échographie (us_A).

---

## Vérifications après chaque téléchargement

```bash
# 1. Le manifeste voit-il le dataset ?
python -m src.data.build_manifest

# 2. Effectifs attendus :
#    busi           ~647 images (780 - 133 normales)
#    cbis_ddsm      ~1644 cas (753 calc + 891 masses)
#    cmmd           ~3712 images / 1775 patientes, avec âges
#    breast_poland  ~252 images (256 - 4 normales)
#
# 3. Refaire les splits :
python -m src.data.splits --manifest data/manifest.csv

# 4. Tout doit rester vert :
python -m pytest tests/ -q
```

Pour CBIS-DDSM et CMMD (DICOM), la conversion se fait ensuite :
```bash
python -m src.data.dicom_convert --dataset cbis_ddsm --limit 20   # test visuel d'abord !
```
Regarder les 20 PNG produits **à l'œil** avant de lancer la conversion complète.
