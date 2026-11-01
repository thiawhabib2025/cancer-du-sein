# Protocole d'évaluation — Piste B (fusion tri-modale sur appariement virtuel)

**Décision d'équipe du 8 août 2026 :** la piste B est l'architecture principale
du projet, avec variables cliniques **C0 uniquement** (âge et type de lésion
réels disponibles dans CMMD). Le type de lésion peut être associé au diagnostic ;
son association doit être discutée et ne constitue pas, à elle seule, une
preuve de fuite.

Ce document est le contrat d'évaluation. Toute expérience de fusion qui ne le
respecte pas ne peut pas figurer dans le rapport.

---

## 1. Ce que « appariement virtuel » veut dire — et ce qu'on peut en conclure

Chaque paire associe une mammographie CMMD et une échographie BUSI de
**deux femmes différentes**, tirées indépendamment des labels dans chaque
split. La cible supervisée est le diagnostic mammographique. Il n'existe donc
**aucune interaction inter-modale réelle** dans les données : l'écho ne
correspond pas à la patiente de la mammographie.

Conséquences à garder en tête :

- On **peut** décrire : les performances des architectures dans ce protocole
  artificiel, la variabilité due aux cinq appariements, et la sensibilité aux
  échographies permutées lors du contrôle C4.
- On **ne peut pas** conclure sur : le gain clinique réel d'une fusion
  mammo+écho chez une vraie patiente. Toute phrase du rapport qui le
  suggérerait est à réécrire.

Dans le rapport et à la soutenance, parler de **« paires virtuelles
inter-cohortes, tirées indépendamment du label »**. L'appariement conditionné
par le label est interdit en validation et en test, car il exploiterait la
cible au moment de construire les paires.

## 2. Les cinq modèles comparés

| Modèle | Rôle |
|---|---|
| `logit_averaging` | **Baseline obligatoire** — aucune interaction modélisée, 3 poids appris |
| `early` | Borne basse attendue — fusion au niveau pixel |
| `late` | Référence de la littérature (Chen et al. 2025) |
| `cross_attention` | Le mécanisme dont on évalue l'utilité |
| `cross_attention_amw` | + pondération par cas (Rashedi et al. 2026) |

Auxquels s'ajoutent les 3 mono-modaux de la Phase 2 : mammo seule, écho seule,
clinique seule (MLP sur C0).

**Règle :** même architecture d'extracteur (`resnet18` pré-entraîné) pour les
cinq méthodes. Les poids sont ajustés pendant chaque entraînement ; ils ne
sont pas figés. Chaque graine d'appariement repart de la même initialisation
pré-entraînée, les cinq graines ne sont pas des initialisations indépendantes.

## 3. Les quatre comparaisons obligatoires (dans cet ordre)

### C1. Fusion vs références unimodales sur la même cohorte
Comparer la fusion sélectionnée sur validation à la mammographie seule et à la
clinique seule, toutes deux évaluées sur les mêmes patientes CMMD. La baseline
échographique BUSI est une cohorte distincte et ne peut pas être classée contre
les résultats CMMD. Les p-values sont corrigées par Holm dans la famille de
comparaisons exportée.

### C2. Cross-attention vs moyenne des logits ⭐ LA comparaison du projet
C'est elle qui répond à la question de recherche. Trois issues possibles :

| Issue | Interprétation | Valeur pour le rapport |
|---|---|---|
| CA > LogitAvg (après correction) | Le mécanisme capte une différence prédictive dans ce protocole ; cela ne démontre pas une interaction clinique réelle. Examiner C4. | Résultat à expliquer avec prudence |
| CA ≈ LogitAvg | Aucune différence prédictive détectée par ce protocole par rapport à la référence. | Résultat descriptif ; ne pas généraliser à une fusion clinique réelle |
| CA < LogitAvg | Le modèle est moins performant dans ce protocole. | Résultat à interpréter sans attribuer de cause non testée |

Les trois issues font un bon mémoire. Aucune n'est un échec.

### C3. AMW vs cross-attention simple
Comparer les deux méthodes sur les mêmes patientes avec DeLong. Les poids de
modalité AMW sont descriptifs de ce modèle et de ces paires virtuelles ; ils ne
mesurent pas une contribution causale ou une importance clinique.

### C4. Contrôle d'intégrité : permutation à l'inférence
Après sélection de la stratégie sur la validation, conserver les cinq modèles
entraînés et permuter les échographies du hold-out sans modifier les
mammographies, labels ou identifiants. Recalculer les prédictions agrégées
sans réentraînement. Ce contrôle vérifie la sensibilité du modèle à l'entrée
échographique ; il ne constitue pas un nouveau modèle ni une preuve de gain
clinique.

## 4. Variance d'appariement : 5 tirages

Chaque stratégie est entraînée avec `pair_seed ∈ {0, 1, 2, 3, 4}`. On rapporte
**moyenne ± écart-type de l'AUC par tirage**, ainsi que l'évaluation secondaire
des prédictions moyennées des cinq tirages. Les métriques de seuil et les tests
de DeLong portent sur ces prédictions moyennées ; le seuil est choisi sur leurs
scores de validation moyennés.

Lecture : si l'écart-type entre tirages est du même ordre que l'écart entre
stratégies, les différences observées relèvent du hasard d'appariement, pas des
architectures. C'est le second garde-fou après DeLong.

## 5. Budget

5 stratégies × 5 tirages + 3 baselines = **28 entraînements** ; le contrôle
par permutation est une inférence supplémentaire, soit **29 expériences**.
Le notebook utilise un train constitué des folds 0–3, le fold 4 comme
validation, et le hold-out préétabli (`fold == -1`) une seule fois à la fin.
Les cinq graines mesurent la variabilité de l'appariement, pas cinq splits
patients ni cinq initialisations indépendantes.

Ordre d'exécution prioritaire : les trois baselines, puis `logit_averaging`,
`late`, `cross_attention` et `cross_attention_amw` avec cinq graines chacune ;
`early` complète la comparaison. Si le temps manque, l'option du notebook
permet d'omettre `early`, mais le rapport doit alors l'indiquer comme non
exécutée. Les temps d'exécution doivent être mesurés sur le runtime choisi ;
aucune durée GPU ne doit être annoncée avant mesure.

## 6. Formulations pour le rapport

À utiliser :
> « En l'absence de jeu public appariant les deux modalités chez la même
> patiente, nous évaluons des paires virtuelles inter-cohortes tirées sans
> utiliser les labels pour apparier. Les résultats ne mesurent pas un bénéfice
> clinique de fusion chez une patiente réelle. »

À proscrire :
> « Notre modèle fusionne la mammographie et l'échographie de la patiente »
> (faux : ce ne sont pas les images d'une même patiente).
