# CS Pole IA - S6.07

## Organisation

Le présent code se divise en 3 sections:
- basic contient un simple ResNet18, sans méthode de Sample Selection
- dividemix contient une implémentation de ResNet18, implémenté avec l'algorithme de Sample Selection DivideMix
- fine contient une implémentation de ResNet (architecture modulable), implémenté avec l'algorithme de Sample Selection FINE, et certaines de ses variantes (voir le README de cette section pour obtenir des détails supplémentaires)

## Utilisation 

Pour en faire usage, le notebook Kaggle ici lié permet de réaliser l'ensemble des imports nécessaires. La selection du code à éxécuter se fait:
-dans le cas de DivideMix, ...
-dans le cas de FINE, en se plaçant dans le répertoire fine, puis dans le sous-répertoir désiré : fine_dividemix, ou dynamic_selection pour toute autre méthode (détails dans les README concernés).


Si le lecteur désire utiliser les méthodes sans employer ce notebook:
-la version de python doit être 3.7
-nous recommandons d'employer les commandes suivantes (ou équivalent) pour répliquer les conditions employés dans notre rapport. Nous ne garantissons pas l'exécution du code avec d'autres versions des librairies concernées.

```

!pip uninstall -y torchvision requests tqdm torch mlflow Pillow scikit_learn wandb torchnet

!pip install torchvision==0.9.0 requests==2.23.0 tqdm==4.28.1 torch==1.8.0 mlflow==1.11.0 Pillow==8.1..0 scikit_learn==0.24.0 wandb torchnet

!pip install --upgrade protobuf==3.20.0

!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

!pip install matplotlib


```


Veuillez alors vous placer dans le répertoire désiré, puis vous réferer au REAMDE de la section concernée.





####

Implémentations de plusieurs méthodes :
+ basic : Sans méthodes particulières
Résultats ici : https://www.kaggle.com/code/eternel22/poleia-basic?scriptVersionId=131912917
+ fine : Paper, NeurIPS 21, <i>FINE Samples for Learning with Noisy Labels</i>, [[Arxiv](https://arxiv.org/abs/2102.11628)][[OpenReview](https://openreview.net/forum?id=QZpx42n0BWr)]
+ dividemix : <i>DivideMix: Learning with Noisy Labels as Semi-supervised Learning</i> <a href="https://openreview.net/pdf?id=HJgExaVtwr">[pdf]</a>

Pour déterminer les scores, on a entraîné un Resnet sur des données non bruités : 
https://www.kaggle.com/code/eternel22/poleia-basic-bruitage?scriptVersionId=131835320
*

#####
