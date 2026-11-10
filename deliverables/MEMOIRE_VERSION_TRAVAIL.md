# Fusion multimodale pour la détection du cancer du sein

## Mémoire — version de travail

**Projet de Master 1 ILIA — École Polytechnique de Thiès**  
**Équipe :** Fallou Diouck, Habib Thiaw, Abibatou Sylla  
**Version :** 1 novembre 2026  
**Statut :** document de travail. Les résultats expérimentaux historiques présentés ci-dessous sont exploratoires et ne constituent pas une évaluation indépendante ni une validation clinique.

> **Consigne de révision.** Les champs `[À COMPLÉTER APRÈS VALIDATION]` doivent être remplacés uniquement après une exécution reproductible du protocole final. Ne pas supprimer les avertissements de statut tant que les résultats n’ont pas été revérifiés.

## Résumé provisoire

Ce mémoire étudie la classification de lésions mammaires à partir d’images et de variables cliniques observées, ainsi que l’intérêt de stratégies de fusion. Le périmètre opérationnel documenté dans le dépôt associe les mammographies CMMD aux échographies BUSI et aux variables réellement disponibles dans CMMD, principalement l’âge et le type d’anomalie. Les deux jeux d’images ne contiennent pas d’examens mammographiques et échographiques appariés chez les mêmes patientes. La fusion inter-cohortes est donc virtuelle et ne mesure pas un bénéfice clinique chez une patiente réellement examinée dans les deux modalités.

Le code comprend des composants de préparation des données, des séparations groupées par identifiant patient lorsque celui-ci est exploitable, des architectures unimodales et des stratégies de fusion, ainsi qu’un notebook expérimental. Des sorties historiques exportées sur Drive rapportent notamment des AUC-ROC de 0,9550 pour la moyenne des logits et de 0,9354 pour la cross-attention avec AMW. Ces nombres sont **exploratoires** : l’exécution correspondante utilise une validation dont l’appariement échographie–mammographie est conditionné par le diagnostic, ne présente pas de hold-out indépendant, et ses comparaisons statistiques ne garantissent pas l’alignement patient nécessaire. Ils ne sont donc pas interprétés comme performances finales.

La version actuelle du protocole prévoit un appariement indépendant des labels en validation et au test, une sélection sur validation, une évaluation finale au niveau patient et des comparaisons statistiques contrôlées. Cette version doit encore être exécutée sur un environnement compatible PyTorch/timm. Le présent document décrit l’état établi, les traces exploratoires et les vérifications requises avant toute conclusion quantitative.

**Mots-clés :** cancer du sein, mammographie, échographie, fusion multimodale, apprentissage profond, appariement virtuel, explicabilité.

## 1. Introduction

Le cancer du sein est diagnostiqué à partir d’informations complémentaires : imagerie, caractéristiques de la lésion et contexte clinique. La mammographie et l’échographie peuvent fournir des vues distinctes d’une anomalie, tandis que des variables cliniques peuvent contribuer à la stratification. L’apprentissage profond permet d’extraire des représentations d’images et de comparer des architectures de classification.

Une fusion multimodale fiable nécessite toutefois des exemples correctement associés : les modalités doivent représenter les mêmes patientes, avec des labels cohérents. Les jeux publics retenus ici ne satisfont pas cette condition entre CMMD et BUSI. Cette contrainte modifie la question scientifique : le projet peut comparer des modèles dans un protocole de paires virtuelles inter-cohortes, mais il ne peut pas démontrer que l’échographie améliore le diagnostic d’une patiente donnée lorsque la mammographie de cette même patiente est disponible.

### 1.1 Question de recherche

Dans quelle mesure différentes stratégies de combinaison de représentations d’imagerie et de variables cliniques observées se comportent-elles dans un protocole expérimental inter-cohortes, par rapport à des baselines unimodales et à une combinaison sans interaction explicite ?

Cette formulation ne présuppose ni gain de performance, ni bénéfice clinique.

### 1.2 Objectifs

1. Décrire et vérifier le pipeline de données et les variables effectivement disponibles.
2. Comparer les baselines mammographie, échographie et clinique aux méthodes de fusion prévues.
3. Mesurer la variabilité liée aux tirages d’appariement virtuel.
4. Évaluer les prédictions et l’explicabilité avec des métriques définies avant d’interpréter les résultats.
5. Présenter un prototype logiciel de recherche sans le confondre avec un dispositif médical.

## 2. Contexte et travaux connexes

Le cahier des charges et les documents d’orientation décrivent des travaux sur les réseaux convolutifs et les Transformers pour la classification d’images mammaires, ainsi que sur les fusions précoce, tardive et par attention. Les méthodes retenues dans le protocole comprennent une moyenne pondérée des logits, des concaténations et une cross-attention avec ou sans pondération adaptative des modalités.

La contrainte centrale du projet est l’absence de correspondance patient entre les sources d’imagerie disponibles. Les paires virtuelles doivent donc être interprétées comme une expérience méthodologique artificielle. Le modèle apprend à prédire le label mammographique à partir d’une entrée mammographique, d’une échographie provenant d’une autre observation et, lorsque disponible, de variables cliniques CMMD. Une éventuelle différence entre méthodes ne constituerait pas une preuve d’interaction clinique réelle.

**Références bibliographiques à compléter :** reprendre et vérifier les références détaillées du cahier des charges et des rapports d’orientation. Les références associées aux architectures dans `model_params.csv` ne sont pas suffisantes, à elles seules, pour établir les résultats ou les configurations utilisés dans les entraînements.

## 3. Données et variables

### 3.1 Périmètre opérationnel

Les effectifs suivants proviennent du manifeste local et de la synthèse Drive consultée. Ils décrivent les données, pas la qualité d’un modèle.

| Jeu | Modalité | Effectif documenté | Variables/limites |
|---|---|---:|---|
| CMMD | Mammographie | 3 744 images, 1 775 identifiants patient | Âge réel et type d’anomalie selon le manifeste ; le diagnostic est la cible |
| BUSI | Échographie | 647 images | Masques disponibles pour l’analyse de localisation ; identifiant de patiente non exploitable de manière fiable dans le pipeline retenu |
| Total | Deux cohortes séparées | 4 391 images | Les modalités ne sont pas appariées chez les mêmes patientes |

Le manifeste CMMD permet un découpage groupé par patiente. Pour BUSI, traiter une image comme un groupe distinct ne garantit pas que des vues d’une même personne soient séparées entre plis. Les métriques des deux cohortes ne doivent pas être comparées comme si elles portaient sur une population commune.

### 3.2 Prévalence et risques de raccourcis

Les sorties historiques indiquent une proportion positive d’environ 70 % dans CMMD et d’environ 32 % dans BUSI. Les différences de prévalence, de population, de matériel et de protocole d’acquisition peuvent créer des raccourcis liés à la provenance des images. Une analyse inter-cohortes ne permet pas, à elle seule, d’isoler l’effet de la modalité.

Les variables cliniques synthétiques ne sont pas admises dans les résultats principaux. Le protocole retient les informations réellement observées. Le type d’anomalie peut néanmoins être associé au diagnostic : son rôle et sa disponibilité doivent être décrits, et le score de la baseline clinique doit faire l’objet d’un audit spécifique.

### 3.3 Prétraitement documenté dans le dépôt

Le fichier `configs/data.yaml` décrit une taille d’image de 224 × 224, une conversion en RGB pour les encodeurs pré-entraînés et la normalisation ImageNet (moyennes 0,485/0,456/0,406 ; écarts-types 0,229/0,224/0,225). Le pipeline DICOM prévoit l’application des LUT de visualisation. Les augmentations documentées comprennent le retournement horizontal, une rotation limitée et des variations photométriques ; le retournement vertical est désactivé.

Ces paramètres décrivent la configuration versionnée. Ils ne prouvent pas à eux seuls que les checkpoints Drive ont été entraînés exactement avec ces paramètres.

## 4. Méthodologie expérimentale

### 4.1 Séparation des données

La règle méthodologique est de séparer les patientes, et non les images, lorsque des identifiants fiables existent. Le protocole versionné réserve les patientes `fold = -1` au hold-out, utilise les folds 0 à 3 pour l’entraînement et le fold 4 pour la validation. Le hold-out ne doit servir qu’à l’évaluation finale après sélection de la stratégie et du seuil.

**État vérifié :** ce schéma est documenté dans les sources du dépôt. La nouvelle exécution complète et l’intégrité effective de toutes ses sorties restent à établir.

### 4.2 Modèles prévus

Les trois références unimodales sont mammographie seule, échographie seule et clinique seule. Les méthodes multimodales définies dans le protocole sont :

| Méthode | Fonction dans la comparaison |
|---|---|
| Moyenne des logits (`logit_averaging`) | Référence sans interaction explicite |
| Fusion précoce (`early`) | Combinaison en entrée du réseau |
| Fusion tardive (`late`) | Combinaison des représentations avant la tête de classification |
| Cross-attention (`cross_attention`) | Attention entre représentations |
| Cross-attention + AMW | Pondération adaptative des modalités |

Les extracteurs comparés dans les documents comprennent ResNet18, EfficientNet-B7, ReXNet-200, ViT-B/16 et Swin-Tiny. La comparaison principale de fusion prévue utilise ResNet18 afin de limiter les différences d’encodeur. La résolution native d’EfficientNet-B7 est de 600 pixels dans le registre du code ; une entrée à 224 pixels doit être signalée comme un compromis si cette architecture est évaluée.

### 4.3 Appariement virtuel

Les deux images d’une paire proviennent de jeux et de patientes différents. L’appariement doit être reproductible et indépendant des labels lors de la validation et du test. Le protocole complet prévoit cinq tirages d’appariement par stratégie de fusion ; ces graines mesurent la variabilité de l’appariement, pas cinq initialisations indépendantes ni cinq cohortes indépendantes.

Le budget prévu est de 3 baselines + 5 stratégies × 5 tirages = 28 entraînements, auxquels s’ajoute un contrôle par permutation des échographies à l’inférence, sans réentraînement, soit 29 expériences au sens du plan. Cette exécution complète n’est pas démontrée par les fichiers examinés.

### 4.4 Métriques et tests

Le protocole prévoit AUC-ROC, AUC-PR, accuracy, sensibilité, spécificité, précision, F1 et MCC. Les métriques de seuil doivent utiliser un seuil choisi sur la validation. Les scores doivent être agrégés par patiente avant les comparaisons dans CMMD. Les comparaisons d’AUC doivent partager exactement les patientes, les labels et leur ordre ; les comparaisons multiples prévues requièrent une correction de Holm.

**Emplacements à compléter après validation :**

- **Cohorte et hold-out final :** [À COMPLÉTER APRÈS VALIDATION]
- **Effectifs patients et classe par partition :** [À COMPLÉTER APRÈS VALIDATION]
- **Seuils choisis exclusivement sur validation :** [À COMPLÉTER APRÈS VALIDATION]
- **Métriques patientes avec intervalles de confiance :** [À COMPLÉTER APRÈS VALIDATION]
- **Comparaisons DeLong alignées et p-values corrigées par Holm :** [À COMPLÉTER APRÈS VALIDATION]
- **Variabilité moyenne ± écart-type des cinq appariements :** [À COMPLÉTER APRÈS VALIDATION]

## 5. Implémentation et reproductibilité

Le dépôt contient des modules de données, de prétraitement, de modèles, d’entraînement, d’évaluation et d’explicabilité, ainsi que des tests portant notamment sur la séparation des patients, les métriques et la reprise d’entraînement. Le notebook versionné prévoit une sauvegarde de checkpoints et des exports ; sa syntaxe et la structure JSON ont été vérifiées dans un audit précédent.

L’environnement local examiné ne dispose pas de PyTorch ni de timm. Les tests qui nécessitent les réseaux et leurs passes avant/arrière n’y valident donc pas l’exécution des modèles. Les checkpoints accessibles sur Drive n’ont pas été téléchargés et inspectés dans le présent environnement ; leur architecture, prétraitement, empreinte et compatibilité avec le code ne sont pas confirmés.

La présentation actuelle des résultats historiques provient d’un notebook exécuté sur un runtime T4. Le code de cette exécution apparie les exemples de validation en fonction du diagnostic, contrairement au protocole versionné. Ce fait est déterminant pour le statut des chiffres ci-dessous.

## 6. Résultats disponibles

### 6.1 Résultats historiques — exploratoires, non validés

Les valeurs du tableau proviennent des exports Drive et sorties du notebook historique. Elles sont rapportées pour assurer la traçabilité, **pas comme performances finales**. Le jeu BUSI et le jeu CMMD sont différents ; leurs lignes ne forment pas une comparaison appariée.

| Modèle | Cohorte | n documenté | AUC-ROC exploratoire | AUC-PR exploratoire |
|---|---|---:|---:|---:|
| Échographie seule | BUSI | 112 | 0,9339 | 0,8983 |
| Mammographie seule | CMMD | 634 | 0,7528 | 0,8769 |
| Clinique seule | CMMD | 634 | 0,7462 | 0,8697 |
| Moyenne des logits | CMMD | 634 | 0,9550 | 0,9776 |
| Cross-attention + AMW | CMMD | 634 | 0,9354 | 0,9619 |
| AMW, contrôle à appariement aléatoire | CMMD | 634 | 0,7844 | 0,8946 |

Ces effectifs sont ceux des sorties évaluées par image, et non des comptes d’unités patientes validés dans un hold-out indépendant. Les scores au seuil rapportés dans la synthèse historique sont aussi exploratoires ; le seuil de Youden a été calculé sur les données évaluées, ce qui ne fournit pas une estimation indépendante des performances de seuil.

### 6.2 Pourquoi ces résultats ne sont pas finaux

1. **Appariement conditionné par la cible :** le `PairDS` historique forme des paires de même diagnostic en validation. Le label mammographique influence alors l’échographie reçue par le modèle.
2. **Pas de hold-out final démontré :** les sorties historiques excluent le fold `-1`, utilisent un fold comme validation et d’autres folds pour l’entraînement. Elles ne fournissent pas l’évaluation finale annoncée par le protocole révisé.
3. **Comparaisons statistiques désalignées :** le code historique teste la longueur des vecteurs pour DeLong, pas l’identité et l’ordre des patientes et des labels. Les AUC enregistrées dans `delong.csv` pour certaines références diffèrent des AUC de `metriques.csv`. Le fichier examiné ne documente pas la correction de Holm.
4. **Agrégation :** les exports historiques sont par image ; une métrique par patiente n’est pas établie.
5. **Baseline clinique à investiguer :** AUC exploratoire 0,7462, au-dessus du seuil de contrôle 0,65 mentionné dans les documents. Le lien entre type d’anomalie et diagnostic ainsi que la construction des variables doivent être réexaminés.
6. **Contrôle d’appariement :** l’AUC exploratoire du contrôle aléatoire ne démontre ni l’utilité ni l’inutilité d’une modalité sans une évaluation alignée et sans cible indépendante.

### 6.3 Visualisations et explicabilité — exploratoires

La figure ROC exportée compare visuellement la moyenne des logits, la cross-attention avec AMW et les trois références unimodales. Sa légende rapporte les AUC du tableau ci-dessus. Elle illustre les sorties de l’évaluation historique, mais hérite des limites de cette évaluation et ne doit pas être présentée comme une courbe de test indépendante.

Le résumé Grad-CAM Drive rapporte, pour 112 images BUSI, un IoU moyen de 0,1101, une médiane de 0 et un pointing game de 0,2232. Une autre sortie du notebook historique rapporte IoU moyen 0,1029 sur 112 images. Cette divergence doit être résolue en retraçant la version de code et le fichier exporté avant toute citation unique. La figure Grad-CAM montre des exemples de recouvrement variable ; certains sont visuellement plausibles, tandis que d’autres cartes répondent au contexte ou aux bords.

SHAP a échoué dans l’exécution rapportée (« # of weights must match data matrix »). Aucun résultat SHAP valide ne peut être présenté. Les poids appris par la moyenne des logits sont descriptifs du modèle et ne représentent pas une contribution causale ou une importance clinique.

**Emplacements finaux :** [À COMPLÉTER APRÈS VALIDATION : ROC/PR du hold-out] ; [À COMPLÉTER : matrices de confusion avec seuils de validation] ; [À COMPLÉTER : courbes d’apprentissage] ; [À COMPLÉTER : localisation réévaluée] ; [À COMPLÉTER : SHAP, si le calcul est corrigé et contrôlé].

## 7. Discussion

Les sorties exploratoires montrent une AUC élevée pour la moyenne des logits, mais elles ne permettent pas d’attribuer cet écart à une complémentarité clinique entre mammographie et échographie. La construction des paires historiques expose le modèle de fusion au diagnostic lors de la sélection de l’échographie en validation. La différence de cohorte et la forte AUC clinique demandent également un audit avant l’analyse du comportement des modèles.

La démarche révisée est plus prudente : appariement indépendant du label en validation et test, sélection de modèle sur validation, hold-out réservé, comparaisons par patiente et corrections statistiques. Même si cette procédure produit un résultat robuste, l’absence d’images réellement appariées empêchera toujours de conclure à un bénéfice clinique de la fusion chez une patiente individuelle.

Les résultats Grad-CAM disponibles ne démontrent pas une localisation fiable de la lésion. Ils doivent être présentés avec les statistiques globales, pas uniquement par une sélection d’images. La métrique de localisation dépend aussi de la qualité et de la définition des masques de référence.

## 8. Prototype d’interface

Le dépôt ne contenait pas d’interface avant cette version du mémoire. Une interface minimale de démonstration est en cours de préparation séparément. Aucun checkpoint local n’est disponible dans l’espace de travail examiné et les bibliothèques PyTorch/timm sont absentes ; aucune inférence ne peut donc être validée à cette étape.

Le prototype doit rester désactivé pour les scores tant qu’un checkpoint n’a pas été associé à une architecture, un prétraitement et une configuration vérifiables, puis chargé strictement dans un environnement compatible. L’application n’est pas un dispositif médical, n’a pas été validée cliniquement et ne doit pas être utilisée pour le diagnostic.

## 9. Limites et travaux futurs

1. Absence de paires mammographie–échographie chez les mêmes patientes.
2. Identifiants BUSI insuffisants pour garantir une séparation par patiente.
3. Différences de prévalence, de population et d’acquisition entre CMMD et BUSI.
4. Résultats historiques non indépendants, appariement dépendant du diagnostic et mesures par image.
5. Désalignement des comparaisons DeLong historiques et correction multiple non documentée.
6. Score clinique historique nécessitant une analyse des variables et du codage.
7. Localisation faible et divergente entre fichiers ; SHAP non disponible.
8. Checkpoints Drive non inspectés et non testés dans le code local.
9. Absence d’évaluation externe indépendante, de validation prospective et d’étude utilisateur clinique.

Travaux prioritaires : valider les fichiers et métadonnées des checkpoints ; exécuter le protocole corrigé avec un runtime adapté ; vérifier l’agrégation et l’alignement patient ; publier les exports complets ; compléter les analyses d’explicabilité ; réaliser une démonstration technique sans allégation clinique.

## 10. Conclusion provisoire

Le projet dispose d’un pipeline logiciel et d’un protocole expérimental documentés, ainsi que de résultats historiques qui attestent qu’une partie des modèles a été entraînée. Ces résultats sont exploratoires et ne satisfont pas les conditions nécessaires à une estimation finale de généralisation. La contribution défendable à ce stade est la formalisation explicite des limites d’appariement et la préparation d’une comparaison contrôlée contre une baseline sans interaction. Toute conclusion sur les performances finales devra attendre la nouvelle évaluation.

### Champs obligatoires avant dépôt final

- [À COMPLÉTER : date, version du code et empreintes des données]
- [À COMPLÉTER : nombre exact d’expériences réellement exécutées]
- [À COMPLÉTER : effectifs par patiente et par partition]
- [À COMPLÉTER : tableau final, métriques par patiente et intervalles de confiance]
- [À COMPLÉTER : tests statistiques alignés avec correction de Holm]
- [À COMPLÉTER : validation des checkpoints utilisés]
- [À COMPLÉTER : figures et explications finales cohérentes avec les exports]
- [À COMPLÉTER : références bibliographiques vérifiées et harmonisées]

## Sources de travail consultées

- `README.md` du dépôt `breast-multimodal-fusion`
- `docs/protocole_piste_B.md` et `configs/data.yaml`
- `src/data/datasets.py`, `src/models/encoders.py`, `src/models/fusion.py`, `src/train/trainer.py`
- `RAPPORT_ETAT_PROJET.md`, `GUIDE_SOUTENANCE.md` et rapports d’orientation actualisés
- Notebook Colab historique `04_resultats_presentation.ipynb` et exports Drive `resultats_soutenance`
- `Soutenance_Fusion_MultiModale_actualisee.pptx`
