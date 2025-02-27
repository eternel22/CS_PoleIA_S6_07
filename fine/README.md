# FINE Samples for Learning with Noisy Labels

## Organisation
Cette section se divise en deux partie : 

-dynamic_selection, qui permet d'entrainer des modèles avec FINE , F-Coteaching et FINE-dynamic.
-fine-dividemix, qui permet d'entrainer des modèles avec F-DivideMix

Dans le contexte du projet, une fonction de bruit particulière, détaillée dans le rapport, a été demandée. Celle-ci n'a été implémentée pour le moment que dans la section dynamic_selection, pour le dataloader Cifar-10, partagé par FINE, FINE-Dynamic (combiné à des fonction à perte robuste classqiues) et F-coteaching. Son implémentation pour F-DivideMix est actuellement en cours.

## Utilisation
Pour exécuter un entrainement, un fois dans la section désirée, appeller : 

!bash scripts/"script désiré"

ou "script désiré vaut : 

```

f-coteaching.sh pour F-Coteaching

/sample_selection_based/fine-cifar_X.sh, où X un taux de bruitage de 00,20,40,60 ou 80%.

/sample_selection_based/refinement_dynamic_c10.sh pour F-DivideMix

/robust_loss/fine_dynamic_X.sh où X vaut gce, sce, elr, ce où les acronymes indiquent les fonctions avec perte robuste Cross Entropy (CE), Generalized Cross Entropy (GCE), Symmetric Cross Entropy (SCE), et Early-Learning Regularized (ELR).

```

Dans tous les cas, les architectures par défaut sont ResNet18, et les taux de bruitage sont de 40%.

Sous dynamic_selection, si le lecteur souhaite exécuter avec un taux de bruitage particulier, ou exécuter avec ResNet34, il peut modifier les arguments "percent X", "et arch rnX".

Sous fine-dividemix, si le lecteur souhaite exécuter avec un taux de bruitage particulier, il peut modifier l'argument noise_mode "sym --r X"

## Commentaires

L'architecture de la section est entièrement reprise de celle du GitHub officiel de FINE. Des fichiers ont cependant été modifiées :

-des erreurs empéchant l'exécution de l'entrainement on été corrigées sur main.py et dynamic_selection/data_loader/cifar10.py.

-le bruitage désiré a été inclus dans le dataloader.

-certains fichiers sans utilisation possible dans le cadre du projet on été supprimés, pour clarifier l'arboresence.

Dans son état actuel, le modèle prédictif requis pour le bruitage est entraîné à plusieurs reprises, ce qui est inutile et représente une part importante du temps d'éxécution du code. Des pistes de solution sont à l'étude.
Une conséquence de ce problème est limitation de son nombre d'époques à 1, ce qui nuit à la qualité du bruitage. Cependant, une machine plus puissante, ou un code amélioré, trouveront en ligne 157 de dynamic_selection/data_loader/cifar10.py ce nombre, qu'ils pourront augmenter pour accroître les performances.
