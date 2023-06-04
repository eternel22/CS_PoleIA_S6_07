# CS Pole IA - S6.07

Le Groupe 7 du projet de S6 du Pôle IA de CentraleSupélec a permis en juin 2023 de reprendre l'implémentation de plusieurs méthodes de Sample Selection, ainsi que la comparaison de ces différentes méthodes.


## Organisation

Le présent code se divise en 3 sections:
- basic contient un simple ResNet18, sans méthode de Sample Selection
- dividemix contient une implémentation de ResNet18, implémenté avec l'algorithme de Sample Selection DivideMix
- fine contient une implémentation de ResNet (architecture modulable), implémenté avec l'algorithme de Sample Selection FINE, et certaines de ses variantes (voir le README de cette section pour obtenir des détails supplémentaires)

## Utilisation 

Pour en faire usage, les notebooks Kaggle ici présents vous permettrons d'exécuter le code simplement, en copiant sur "Copier le notebook", en prenant garde à activer le GPU, et en modifiant le taux de bruitage si besoin :
- Basic : https://www.kaggle.com/code/eternel22/cs-poleia-s6-p07-basic?scriptVersionId=131970624
- DivideMix : https://www.kaggle.com/code/eternel22/poleia-dividemix?scriptVersionId=131916121
- FINE : https://www.kaggle.com/code/eternel22/poleia-finerobustloss?scriptVersionId=132235616, en se plaçant dans le répertoire fine, puis dans le sous-répertoire désiré : fine_dividemix, ou dynamic_selection pour toute autre méthode (détails dans les README concernés).


Si le lecteur désire utiliser les méthodes sans employer ce notebook:
- la version de python doit être de préférence à 3.7
- nous recommandons d'employer les commandes suivantes (ou équivalent) pour répliquer les conditions employés dans notre rapport. Nous ne garantissons pas l'exécution du code avec d'autres versions des librairies concernées.

```

!pip uninstall -y torchvision requests tqdm torch mlflow Pillow scikit_learn wandb torchnet

!pip install torchvision==0.9.0 requests==2.23.0 tqdm==4.28.1 torch==1.8.0 mlflow==1.11.0 Pillow==8.1..0 scikit_learn==0.24.0 wandb torchnet

!pip install --upgrade protobuf==3.20.0

!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

!pip install matplotlib


```
Veuillez alors vous placer dans le répertoire désiré, puis vous réferer au REAMDE de la section concernée.

## Commentaires

Le code ici présent à été en grande partie forkée des GitHubs officiels des méthodes à étudier. Le lecteur les retrouvera aux liens suivants:
+ https://github.com/Kthyeon/FINE_official, de l'article : Paper, NeurIPS 21, FINE Samples for Learning with Noisy Labels, [[Arxiv](https://arxiv.org/abs/2102.11628)][[OpenReview](https://openreview.net/forum?id=QZpx42n0BWr)]
+ https://github.com/LiJunnan1992/DivideMix, de l'article : DivideMix: Learning with Noisy Labels as Semi-supervised Learning <a href="https://openreview.net/pdf?id=HJgExaVtwr">[pdf]</a>

Ce dépôt est publié sous licence MIT.



