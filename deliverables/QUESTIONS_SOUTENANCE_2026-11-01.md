# Questions probables de soutenance

Réponses préparées à partir des éléments vérifiables du dépôt. Les valeurs
historiques citées restent exploratoires ; aucune réponse ne doit les présenter
comme des performances finales.

## 1. Pourquoi avoir choisi CMMD et BUSI ?

CMMD fournit les mammographies et quelques variables cliniques observées,
notamment l’âge et le type d’anomalie. BUSI fournit des échographies et des
masques utilisables pour une analyse de localisation. Ces sources permettent
d’étudier séparément les deux modalités, mais elles ne contiennent pas
d’examens appariés chez les mêmes patientes.

## 2. Vos images mammographiques et échographiques viennent-elles des mêmes patientes ?

Non. Le protocole forme des paires virtuelles inter-cohortes entre CMMD et
BUSI. Une paire ne représente donc pas une patiente suivie par les deux
modalités. Les expériences ne démontrent pas un bénéfice clinique individuel
de fusion ; elles comparent des stratégies dans un protocole expérimental
artificiel.

## 3. Pourquoi parler de résultats exploratoires ?

Les sorties historiques ont été produites avec des limites qui empêchent une
estimation finale fiable : association de l’échographie au diagnostic dans
certaines paires de validation, absence de hold-out indépendant démontré,
métriques par image et comparaisons statistiques sans preuve d’alignement
patient complet. Les valeurs historiques sont conservées pour la traçabilité,
mais doivent être remplacées ou confirmées par le protocole corrigé.

## 4. Que signifient les AUC historiques citées ?

Ce sont des métriques exploratoires extraites des exports historiques, et non
des estimations finales sur un hold-out patient indépendant. Par exemple,
l’AUC-ROC de 0,9550 pour la moyenne des logits ne prouve pas qu’une fusion
améliore le diagnostic d’une patiente. La différence de cohortes et le
protocole d’appariement limitent son interprétation.

## 5. Comment évitez-vous la fuite de données entre patientes ?

Pour les données avec identifiants exploitables, le code réserve les groupes
patients ensemble et contient des tests de non-recouvrement. Le protocole
prévoit un hold-out séparé (`fold = -1`) et une sélection sur validation avant
l’évaluation finale. BUSI est une limite importante : ses identifiants issus
des noms de fichiers ne garantissent pas que les images corrélées d’une même
patiente restent dans un seul pli.

## 6. Quelles variables cliniques sont utilisées ?

Le périmètre retient seulement les informations réellement présentes dans
CMMD, notamment âge et type d’anomalie selon disponibilité. Les variables
synthétiques ne sont pas utilisées dans les résultats principaux. Le type
d’anomalie peut être associé au diagnostic ; son rôle doit donc être décrit et
la baseline clinique auditée, plutôt que d’être interprétée comme preuve
d’une information indépendante.

## 7. Pourquoi comparer à une moyenne de logits ?

La moyenne des logits est une référence de fusion sans interaction explicite
entre les représentations. Elle permet de tester si des mécanismes plus
complexes comme la cross-attention font mieux dans le protocole évalué. Même
une différence robuste resterait une différence dans les paires virtuelles,
pas la preuve d’une complémentarité clinique réelle.

## 8. Les checkpoints ont-ils été validés ?

Quatre fichiers de checkpoint sont présents localement et leur intégrité
d’archive a été contrôlée. Ils n’ont toutefois pas encore été chargés
strictement dans l’environnement local : PyTorch et timm y sont absents.
L’inspection Drive prévue a été bloquée par l’authentification Colab. Leur
architecture exacte, leur prétraitement et leur compatibilité avec le code
actuel restent à confirmer par chargement sûr puis inférence contrôlée.

## 9. Que fait réellement l’application Streamlit ?

Elle permet de charger séparément une mammographie et une échographie
anonymisées, d’en vérifier le fichier, le format et les dimensions, d’afficher
un aperçu, puis d’effacer les fichiers de la session. Elle n’exécute pas de
modèle, ne calcule aucun score et ne produit pas de Grad-CAM. L’avertissement
de prototype non validé cliniquement reste permanent.

## 10. Pourquoi l’interface n’affiche-t-elle pas de prédiction ?

Afficher une valeur sans modèle chargé et vérifié reviendrait à inventer une
sortie. L’interface échoue volontairement en mode fermé : pas de prédiction
avant identification de l’architecture et du prétraitement, chargement strict
des poids et test d’inférence reproductible.

## 11. Que faudrait-il pour une évaluation finale ?

Il faut d’abord valider les checkpoints et le prétraitement, puis exécuter le
protocole corrigé avec séparation par patiente lorsque l’identifiant est
fiable. La sélection doit être faite sur validation ; le hold-out doit rester
inutilisé jusqu’à l’évaluation finale. Les prédictions, labels, identifiants,
métriques par patiente et tests statistiques corrigés doivent être contrôlés
avant de mettre à jour les documents.

## 12. Quelle est la principale limite et quelle est la suite ?

La limite centrale est l’absence de données publiques appariant les modalités
chez les mêmes patientes, complétée par la faiblesse des identifiants patient
BUSI. La suite est de valider les checkpoints puis, si les données et le
runtime le permettent, d’exécuter le protocole patient-indépendant et de
mettre à jour rapport et présentation exclusivement à partir des nouveaux
exports vérifiés.

## Formulation de conclusion défendable

> Le projet fournit un pipeline de recherche, un protocole explicite sur ses
> limites et une interface de démonstration sans prédiction simulée. Les
> résultats historiques sont exploratoires ; les performances finales et le
> chargement des checkpoints restent à valider.
