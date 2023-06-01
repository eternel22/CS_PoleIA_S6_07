import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import json
import os
import copy

def fix_seed(seed=888):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


#######################################################
#####################################################

#######################################################
#####################################################











#######################################################
#####################################################

def get_cifar10(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True, noise_file = '', teacher_idx=None, seed=888):
  ########

  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib
  import seaborn as sns
  sns.set()
  import os
  import random
  import shutil
  import time
  import warnings
  import numpy as np

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.backends.cudnn as cudnn
  import torchvision
  import torchvision.transforms as transforms 

  from tqdm import tqdm
  from PIL import Image
  import pickle
  from torch.utils.data import DataLoader

  import tarfile
  from torchvision.datasets.utils import download_url
  from torch.utils.data import random_split
  import torch.utils.data as Data
  from torchvision.datasets import ImageFolder
  from torchvision.transforms import ToTensor

  import sklearn
  from sklearn.model_selection import train_test_split
  from sklearn.utils import resample

  import matplotlib
  import matplotlib.pyplot as plt

  # from google.colab import drive
  # drive.mount('/content/drive')

  from urllib.request import urlretrieve

  def unpickle(file):
    import pickle
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
      path = os.path.join('/content','data','cifar10')
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

  train_images, train_labels, val_images, val_labels, test_images, test_labels = cifar10(nbTrainsMax=100, propBruitage=0.5, normalise=True)


  train_images, train_labels, val_images, val_labels, test_images, test_labels = cifar10(nbTrainsMax=1000)


  # Supposons que vous avez un array d'étiquettes 'labels'

  # Calculer les fréquences de chaque label
  frequences = np.bincount(train_labels)

  # Obtenir les labels uniques
  labels_uniques = np.unique(train_labels)

  # Créer un graphique à barres pour représenter l'histogramme de fréquence
  plt.bar(labels_uniques, frequences)

  # Étiqueter les axes
  plt.xlabel('Label')
  plt.ylabel('Fréquence')

  # Afficher le graphique
  plt.show()



  import torch
  import torchvision
  import torch.optim as optim
  import torch.nn as nn
  from torch.utils.data import DataLoader, TensorDataset
  import sys

  train_images, train_labels, val_images, val_labels, test_images, test_labels = cifar10(nbTrainsMax=10000,normalise=True)
  print(type( cifar10(nbTrainsMax=10000,normalise=True)))
    

































    #########
  base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
  if train:
      fix_seed(seed)
      train_idxs, val_idxs = train_labels, val_labels

      train_dataset = TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
      val_dataset = TensorDataset(torch.tensor(val_images), torch.tensor(val_labels))

      cfg_trainer['instance'] = 0
      if cfg_trainer['instance']:
          train_dataset.instance_noise()
          val_dataset.instance_noise()
      elif cfg_trainer['asym']:
          train_dataset.asymmetric_noise()
          val_dataset.asymmetric_noise()
      else:
          train_dataset.symmetric_noise()
          val_dataset.symmetric_noise()
          
      print ('##############')
      print (train_dataset.train_labels[:10])
      print (train_dataset.train_labels_gt[:10])
            
      if teacher_idx is not None:
          print(len(teacher_idx))
          train_dataset.truncate(teacher_idx)
          
      print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
  else:
      fix_seed(seed)
      train_dataset = []
      val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train, transform=transform_val)
      print(f"Test: {len(val_dataset)}")
  
  if len(val_dataset) == 0:
      return train_dataset, None
  else:
      return train_dataset, val_dataset

def train_val_split(base_dataset: torchvision.datasets.CIFAR10, seed):
  fix_seed(seed)
  num_classes = 10
  base_dataset = np.array(base_dataset)
  train_n = int(len(base_dataset) * 0.9 / num_classes)
  train_idxs = []
  val_idxs = []

  for i in range(num_classes):
      idxs = np.where(base_dataset == i)[0]
      np.random.shuffle(idxs)
      train_idxs.extend(idxs[:train_n])
      val_idxs.extend(idxs[train_n:])
  np.random.shuffle(train_idxs)
  np.random.shuffle(val_idxs)

  return train_idxs, val_idxs

class CIFAR10_train(torchvision.datasets.CIFAR10):
  def __init__(self, root, cfg_trainer, indexs, train=True,
                transform=None, target_transform=None,
                download=False, seed=888):
      super(CIFAR10_train, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)
      fix_seed(seed)
      self.num_classes = 10
      self.cfg_trainer = cfg_trainer
      self.train_data = self.data[indexs] # self.train_data[indexs]
      self.train_labels = np.array(self.targets)[indexs] # np.array(self.train_labels)[indexs]
      self.indexs = indexs
      self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
      self.noise_indx = []
      self.seed = seed
              
  def symmetric_noise(self):
      self.train_labels_gt = self.train_labels.copy()
      fix_seed(self.seed)
      indices = np.random.permutation(len(self.train_data))
      for i, idx in enumerate(indices):
          if i < self.cfg_trainer['percent'] * len(self.train_data):
              self.noise_indx.append(idx)
              self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

  def asymmetric_noise(self):
      self.train_labels_gt = copy.deepcopy(self.train_labels)
      fix_seed(self.seed)

      for i in range(self.num_classes):
          indices = np.where(self.train_labels == i)[0]
          np.random.shuffle(indices)
          for j, idx in enumerate(indices):
              if j < self.cfg_trainer['percent'] * len(indices):
                  self.noise_indx.append(idx)
                  # truck -> automobile
                  if i == 9:
                      self.train_labels[idx] = 1
                  # bird -> airplane
                  elif i == 2:
                      self.train_labels[idx] = 0
                  # cat -> dog
                  elif i == 3:
                      self.train_labels[idx] = 5
                  # dog -> cat
                  elif i == 5:
                      self.train_labels[idx] = 3
                  # deer -> horse
                  elif i == 4:
                      self.train_labels[idx] = 7
                      
  def instance_noise(self):
      '''
      Instance-dependent noise
      https://github.com/haochenglouis/cores/blob/main/data/utils.py
      '''
      
      self.train_labels_gt = copy.deepcopy(self.train_labels)
      fix_seed(self.seed)
      
      q_ = np.random.normal(loc=self.cfg_trainer['percent'], scale=0.1, size=int(1e6))
      q = []
      for pro in q_:
          if 0 < pro < 1:
              q.append(pro)
          if len(q)==50000:
              break
              
      w = np.random.normal(loc=0, scale=1, size=(32*32*3, 10))
      for i, sample in enumerate(self.train_data):
          sample = sample.flatten()
          p_all = np.matmul(sample, w)
          p_all[self.train_labels_gt[i]] = -int(1e6)
          p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
          p_all[self.train_labels_gt[i]] = 1 - q[i]
          self.train_labels[i] = np.random.choice(np.arange(10), p=p_all/sum(p_all))
      
  def truncate(self, teacher_idx):
      self.train_data = self.train_data[teacher_idx]
      self.train_labels = self.train_labels[teacher_idx]
      self.train_labels_gt = self.train_labels_gt[teacher_idx]       
      
  def __getitem__(self, index):
      """
      Args:
          index (int): Index

      Returns:
          tuple: (image, target) where target is index of the target class.
      """
      img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img)


      if self.transform is not None:
          img = self.transform(img)

      if self.target_transform is not None:
          target = self.target_transform(target)

      return img, target, index, target_gt

  def __len__(self):
      return len(self.train_data)

class CIFAR10_val(torchvision.datasets.CIFAR10):

  def __init__(self, root, cfg_trainer, indexs, train=True,
                transform=None, target_transform=None,
                download=False):
      super(CIFAR10_val, self).__init__(root, train=train,
                                        transform=transform, target_transform=target_transform,
                                        download=download)

      # self.train_data = self.data[indexs]
      # self.train_labels = np.array(self.targets)[indexs]
      self.num_classes = 10
      self.cfg_trainer = cfg_trainer
      if train:
          self.train_data = self.data[indexs]
          self.train_labels = np.array(self.targets)[indexs]
      else:
          self.train_data = self.data
          self.train_labels = np.array(self.targets)
      self.train_labels_gt = self.train_labels.copy()
      
  def symmetric_noise(self):
      
      indices = np.random.permutation(len(self.train_data))
      for i, idx in enumerate(indices):
          if i < self.cfg_trainer['percent'] * len(self.train_data):
              self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

  def asymmetric_noise(self):
      for i in range(self.num_classes):
          indices = np.where(self.train_labels == i)[0]
          np.random.shuffle(indices)
          for j, idx in enumerate(indices):
              if j < self.cfg_trainer['percent'] * len(indices):
                  # truck -> automobile
                  if i == 9:
                      self.train_labels[idx] = 1
                  # bird -> airplane
                  elif i == 2:
                      self.train_labels[idx] = 0
                  # cat -> dog
                  elif i == 3:
                      self.train_labels[idx] = 5
                  # dog -> cat
                  elif i == 5:
                      self.train_labels[idx] = 3
                  # deer -> horse
                  elif i == 4:
                      self.train_labels[idx] = 7
                      
                      
  def instance_noise(self):
      '''
      Instance-dependent noise
      https://github.com/haochenglouis/cores/blob/main/data/utils.py
      '''
      q_ = np.random.normal(loc=self.cfg_trainer['percent'], scale=0.1, size=int(1e6))
      q = []
      for pro in q_:
          if 0 < pro < 1:
              q.append(pro)
          if len(q)==50000:
              break
              
      w = np.random.normal(loc=0, scale=1, size=(32*32*3, 10))
      for i, sample in enumerate(self.train_data):
          sample = sample.flatten()
          p_all = np.matmul(sample, w)
          p_all[self.train_labels_gt[i]] = -int(1e6)
          p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
          p_all[self.train_labels_gt[i]] = 1 - q[i]
          self.train_labels[i] = np.random.choice(np.arange(10), p=p_all/sum(p_all))
                      
                      
  def __len__(self):
      return len(self.train_data)


  def __getitem__(self, index):
      """
      Args:
          index (int): Index

      Returns:
          tuple: (image, target) where target is index of the target class.
      """
      img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img)


      if self.transform is not None:
          img = self.transform(img)

      if self.target_transform is not None:
          target = self.target_transform(target)

      return img, target, index, target_gt
      
