from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import numpy as np

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

from urllib.request import urlretrieve
import pickle

from dataloader_cifar import cifar10

from torchvision.models import resnet34, ResNet34_Weights


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training - Simple Version')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--bruitage', default=0.0, type=float, help='noise ratio')
parser.add_argument('--seed', default=42)
parser.add_argument('--nbTrains', default=50000)
parser.add_argument('--nbTests', default=10000)
args = parser.parse_args()


def get_net():
    resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) # le parametre preentraine est mis à True
    
    # Substitute the FC output layer
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10) # on change que la dernière couche par rapport au nombre de classes...
    torch.nn.init.xavier_uniform_(resnet.fc.weight) # on initialise cette couche
    return resnet

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

train_images, train_labels, val_images, val_labels, test_images, test_labels = cifar10(nbTrainsMax=args.nbTrains, nbTestsMax=args.nbTests, normalise=True, propBruitage=args.bruitage)

# Préparer les données d'entraînement et de validation
train_dataset = TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_images), torch.tensor(val_labels))

# Définir les dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Définir le modèle, la fonction de perte et l'optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_net()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

# Boucle d'entraînement
num_epochs = args.num_epochs
num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1

for epoch in range(num_epochs):

    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t Train_Loss: %.4f \t Train_Accuracy: %.2f%%'
                %(epoch, num_epochs, i+1, num_iter, loss.item(), correct / total))
        sys.stdout.flush()
    
    history["loss"].append(loss.item())
    history["accuracy"].append(correct / total)
    
    # Évaluation du modèle à la fin de chaque époque
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images.float())
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("")
    print('| Epoch [%3d/%3d] \t Val_Loss: %.4f \t Val_Accuracy: %.2f%%'
                %(epoch, num_epochs, loss.item(), correct / total))
    
    history["val_loss"].append(loss.item())
    history["val_accuracy"].append(correct / total)

# Evaluation
test_dataset = TensorDataset(torch.tensor(test_images), torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
    
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = model(images.float())
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("")
print('| Evaluation : Test_Accuracy: %.2f%%'
            %(correct / total))


def show_results():
    epoch_range = range(1, len(history['loss'])+1)

    plt.figure(figsize=[14,4])
    plt.subplot(1,3,1)
    plt.plot(epoch_range, history['loss'], label='Training')
    plt.plot(epoch_range, history['val_loss'], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(epoch_range, history['accuracy'], label='Training')
    plt.plot(epoch_range, history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()

    plt.savefig('results.png')
    plt.show()

show_results()

torch.save(model.state_dict(), "model")