from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

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
  if tar not in os.listdir(path):
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


class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log='', nbTrains=50000, nbTests=10000): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  

        train_images, train_labels, val_images, val_labels, test_images, test_labels = cifar10(nbTestsMax=nbTests, nbTrainsMax=nbTrains, propBruitage=r, normalise=False)
        _, train_labels_orig, _, _, _, _ = cifar10(nbTestsMax=nbTests, nbTrainsMax=nbTrains, propBruitage=0.0, normalise=False)

        if self.mode=='test':           
            self.test_data = test_images
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  
            self.test_label = test_labels
        
        elif self.mode == "valid":
            self.valid_data = val_images
            self.valid_data = self.valid_data.transpose((0, 2, 3, 1))  
            self.valid_label = val_labels

        else:    
            train_data = train_images
            train_data = train_data.transpose((0, 2, 3, 1))
            train_label = train_labels_orig

            noise_label = train_labels  
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='valid':
            img, target = self.valid_data[index], self.valid_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode=='test':
            return len(self.test_data)
        elif self.mode=='valid':
            return len(self.valid_data)
        else:
            return len(self.train_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='', nbTrains=50000, nbTests=10000):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.nbTrains = nbTrains
        self.nbTests = nbTests
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
            self.transform_valid = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, nbTests=self.nbTests, nbTrains=self.nbTrains)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log,nbTests=self.nbTests, nbTrains=self.nbTrains)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, nbTests=self.nbTests, nbTrains=self.nbTrains)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test',nbTests=self.nbTests, nbTrains=self.nbTrains)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='valid':
            valid_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_valid, mode='valid',nbTests=self.nbTests, nbTrains=self.nbTrains)      
            valid_loader = DataLoader(
                dataset=valid_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return valid_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file,nbTests=self.nbTests, nbTrains=self.nbTrains)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        