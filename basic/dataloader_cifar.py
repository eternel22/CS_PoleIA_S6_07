import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os

from tqdm import tqdm
from PIL import Image
import pickle

import tarfile
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from urllib.request import urlretrieve

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='latin1')
  return dict

def cifar10(nbTrainsMax = 50000, nbTestsMax = 10000, propValid = 0.2, propBruitage = 0, path=None, seed=42, rawCifar=False, normalise=False):
  """
  Cette fonction permet de charger Cifar10, en renvoyant : 
    train_images, train_labels, val_images, val_labels, test_images, test_labels
  
  Paramètres :
    nbTrainsMax : Nombre d'images d'entraînement (train + validation)
    nbTestsMax : Nombre d'images de test
    propValid : Proportion d'images de validation
    propBruitage : Proportion de bruitage entre chats (3) et chiens (5)
    path : path vers cifar10
    seed : seed pour les mélanges
    rawCifar : si on souhaite obtenir cifar sans aucun mélange ni troncature. Dans ce cas, propValid = 0, propBruitage = 0
    normalise : s'il faut normaliser les données
  """

  if path is None:
    path = os.path.join('data','cifar10')
  os.makedirs(path, exist_ok=True)

  # Téléchargement du tar
  url = "https://www.cs.toronto.edu/~kriz/"
  tar = "cifar-10-python.tar.gz"
  if tar not in os.listdir(path):
    urlretrieve(''.join((url, tar)), os.path.join(path, tar))
    print("Downloaded %s to %s" % (tar, path))

  # Téléchargement des scores
  url = "https://github.com/eternel22/CS_PoleIA_S6_07/raw/master/"
  file = "scores_cifar10.npy"
  if file not in os.listdir(path):
    urlretrieve(''.join((url, file)), os.path.join(path, file))
    print("Downloaded %s to %s" % (file, path))
  
  scores = np.load(os.path.join(path, file), allow_pickle=True) # Format : (50 000, 10)

  # Extraction
  with tarfile.open(os.path.join(path, tar), 'r') as fichier_tar:
      fichier_tar.extractall(path=path)
      print("Extraction terminée !")

  path_batches = os.path.join(path, 'cifar-10-batches-py')

  # Création des données de test
  test_dic = unpickle(os.path.join(path_batches, 'test_batch'))
  test_images = test_dic['data']
  test_labels = np.array(test_dic['labels'], dtype=np.uint8)
  
  if not rawCifar:
    test_images, test_labels = resample(test_images, test_labels, n_samples=nbTestsMax, random_state=seed, stratify=test_labels)
  

  # Création des données d'entraînement + validation
  train_images = []
  train_labels = []

  for iBatch in range(1, 6):
    dic = unpickle(os.path.join(path_batches, f'data_batch_{iBatch}'))
    images = dic['data']
    labels = dic['labels']
    train_images.append(images)
    train_labels = train_labels + labels
  
  train_images = np.concatenate(train_images)
  train_labels = np.array(train_labels, dtype=np.uint8)
  val_images = np.array([])
  val_labels = np.array([])

  if not rawCifar:
    train_images, train_labels, scores = resample(train_images, train_labels, scores, n_samples=nbTrainsMax, random_state=seed, stratify=train_labels)
    train_images, val_images, train_labels, val_labels, train_scores, val_scores = train_test_split(train_images, train_labels, scores, test_size=propValid, stratify=train_labels, random_state=seed)

    indices_scores_class_3 = np.where(train_labels  == 3)[0] # tableau 1D contenant les indices des éléments dans train_labels qui ont une valeur de 3.
    indices_scores_class_5 = np.where(train_labels  == 5)[0] # tableau 1D contenant les indices des éléments dans train_labels qui ont une valeur de 5.

    scores_3 = train_scores[indices_scores_class_3, 3] # scores de 3 pour les clean_targets égaux à 3
    scores_5 = train_scores[indices_scores_class_5, 5] # scores de 5 pour les clean_targets égaux à 5

    scores_3_sort = np.argsort(scores_3) # indices des scores 3 triés
    scores_5_sort = np.argsort(scores_5) # indices des scores 5 triés

    indices_scores_class_3_sorted = indices_scores_class_3[scores_3_sort]
    indices_scores_class_5_sorted = indices_scores_class_5[scores_5_sort]

    proportion_to_noisify_3 = int(propBruitage * len(indices_scores_class_3)) 
    proportion_to_noisify_5 = int(propBruitage * len(indices_scores_class_5)) 

    #print(proportion_to_noisify_3)
    #print(proportion_to_noisify_5)

    train_labels_noisy = np.copy(train_labels)

    train_labels_noisy[indices_scores_class_3_sorted[:proportion_to_noisify_3]] = 5
    train_labels_noisy[indices_scores_class_5_sorted[:proportion_to_noisify_5]] = 3
    train_labels = train_labels_noisy


  train_images = train_images.reshape((len(train_images), 3, 32, 32))  
  val_images = val_images.reshape((len(val_images), 3, 32, 32))
  test_images = test_images.reshape((len(test_images), 3, 32, 32)) 

  if(normalise):
    # Normaliser les données d'entrée
    train_images = train_images / 255
    val_images = val_images / 255
    test_images = test_images / 255

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2471, 0.2435, 0.2616])

    train_images = np.transpose(train_images, (0, 2, 3, 1))  # (800, 32, 32, 3)
    train_images = (train_images - mean) / std
    train_images = np.transpose(train_images, (0, 3, 1, 2))  # (800, 3, 32, 32)

    val_images = np.transpose(val_images, (0, 2, 3, 1))  # (800, 32, 32, 3)
    val_images = (val_images - mean) / std
    val_images = np.transpose(val_images, (0, 3, 1, 2))  # (800, 3, 32, 32)

    test_images = np.transpose(test_images, (0, 2, 3, 1))  # (800, 32, 32, 3)
    test_images = (test_images - mean) / std
    test_images = np.transpose(test_images, (0, 3, 1, 2))  # (800, 3, 32, 32)

  return train_images, train_labels, val_images, val_labels, test_images, test_labels

