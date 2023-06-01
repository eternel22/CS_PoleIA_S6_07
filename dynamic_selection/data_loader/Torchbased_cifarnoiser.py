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

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
# from google.colab import drive
# drive.mount('/content/drive')

from urllib.request import urlretrieve

def cifar10(path=None):
  url = "https://www.cs.toronto.edu/~kriz/"
  tar = "cifar-10-binary.tar.gz"
  files = ['cifar-10-batches-bin/data_batch_1.bin',
           'cifar-10-batches-bin/data_batch_2.bin',
           'cifar-10-batches-bin/data_batch_3.bin',
           'cifar-10-batches-bin/data_batch_4.bin',
           'cifar-10-batches-bin/data_batch_5.bin',
           'cifar-10-batches-bin/test_batch.bin']

  if path is None:
	  path = os.path.join(os.path.expanduser('~'),'data','cifar10')

  os.makedirs(path, exist_ok=True)

  if tar not in os.listdir(path):
	  urlretrieve(''.join((url, tar)), os.path.join(path, tar))
	  print("Downloaded %s to %s" % (tar, path))
  
  #urlretrieve(''.join((url, tar)), './cifar-10-binary.tar.tgz')
  with tarfile.open(os.path.join(path, tar)) as tar_object:
    fsize=10000 * (32*32*3) + 10000
    buffr = np.zeros(fsize*6,dtype='uint8')
    members = [file for file in tar_object if file.name in files]
    members.sort(key=lambda member: member.name)

    for i, member in enumerate(members):
      f = tar_object.extractfile(member)
      buffr[i*fsize:(i+1)*fsize] = np.frombuffer(f.read(),'B')
  labels = buffr[::3073]
  pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
  images = pixels.reshape(-1, 3072)#.astype('float32')/255
  train_images, test_images = images[:50000], images[50000:]
  train_labels, test_labels = labels[:50000], labels[50000:]

  def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

  return path, train_images, train_labels, \
        test_images, test_labels

path, train_images, train_labels, test_images, test_labels = cifar10()

class cifar10_noisy(Data.Dataset):
  def __init__(self, scores, root="~/Workspace/noisy_labels/cifar10",
        train=True,
        transform=None,
        transform_eval = None,
        target_transform=None,
        flip_rate_fixed = 0.1,
        random_state = 1):

        path, train_x, train_y,test_x, test_y = cifar10()
        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.t_matrix = None
        root = path
        self.root = root
        self.apply_transform_eval = False

        if train:
            self.images = train_x
            #self.targets = train_y
            self.clean_targets = train_y

            indices_scores_class_3 = np.where(self.clean_targets  == 3)[0]
            indices_scores_class_5 = np.where(self.clean_targets  == 5)[0]

            scores_3 = scores[indices_scores_class_3,3]
            scores_5 = scores[indices_scores_class_5, 5]

            scores_3_sort = np.argsort(scores_3)
            scores_7_sort = np.argsort(scores_5)

            indices_scores_class_3_sorted = indices_scores_class_3[scores_3_sort]
            indices_scores_class_5_sorted = indices_scores_class_5[scores_7_sort]

            proportion_to_noisify_3 = int(flip_rate_fixed * len(indices_scores_class_3)) 
            proportion_to_noisify_5 = int(flip_rate_fixed * len(indices_scores_class_5)) 

            print(proportion_to_noisify_3)
            print(proportion_to_noisify_5)

            y_train_noisy_label= np.copy(self.clean_targets)

            y_train_noisy_label[indices_scores_class_3_sorted[:proportion_to_noisify_3]] = 5
            y_train_noisy_label[indices_scores_class_5_sorted[:proportion_to_noisify_5]] = 3
            self.targets = y_train_noisy_label

        else:
            self.images = test_x
            self.targets = test_y
            self.clean_targets = test_y
            #print('TEEEST', self.targets.shape)
            #self.targets = self.targets.reshape((self.targets.shape[0],)). astype(int)
            #self.clean_targets = self.clean_targets.reshape((self.clean_targets.shape[0],)). astype(int)
        self.images = self.images.reshape((-1,3,32,32))
        self.images = self.images.transpose((0, 2, 3, 1))

        self.test_images = test_x
        self.test_labels = test_y

        self.test_images = self.test_images.reshape((-1,3,32,32))
        self.test_images = self.test_images.transpose((0, 2, 3, 1))

        self.train_images = train_x
        self.train_labels = train_y

        self.train_images = self.train_images.reshape((-1,3,32,32))
        self.train_images = self.train_images.transpose((0, 2, 3, 1))
      
  def __getitem__(self, index):
        img, label, clean_label = self.images[index], self.targets[index], self.clean_targets[index]
        img = Image.fromarray(img)

        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform

        if self.transform is not None:
            img = transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
            clean_label = self.target_transform(clean_label)

        return img, label, clean_label


  def _get_test_images(self):
    return self.test_images, self.test_labels

  def _get_train_images(self):
    return self.train_images, self.train_labels

  def _get_noisy_indices(self):
    indices_noisy_data = []
    for i in range(len(self.clean_targets)):
        if self.clean_targets[i] != self.targets[i]:
            indices_noisy_data.append(i)
    print('nb data', len(self.clean_targets))
    print('nb noisy data', len(indices_noisy_data))
    return indices_noisy_data


  def _set_targets(self,n_targets):
        self.targets = n_targets

  def _get_num_classes(self):
        return len(set(self.targets))

  def _get_targets(self):
        return self.targets

  def eval(self):
        self.apply_transform_eval = True

  def train(self):
        self.apply_transform_eval = False

  def __len__(self):
        return len(self.targets)
def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target
  
  
mean = (0.4915, 0.4823, 0.4468)
std = (0.2470, 0.2435, 0.2616)
normalize = transforms.Normalize(mean, std)
transform_train = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])
transform_test = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained ResNet-34 model
model = models.resnet34(pretrained=True)
model.to(device)

# Replace the last layer with a new fully connected layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)  # 10 classes in CIFAR-10
model.to(device)

# Define the transformation to apply to the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Define the optimizer and loss function for fine-tuning
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the model on the CIFAR-10 training dataset
model.train()
for epoch in range(5):  # Adjust the number of epochs as needed
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 200 == 0:
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/200}")
            running_loss = 0.0

# Download and load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Get the total number of classes in CIFAR-10
num_classes = len(testset.classes)

# Create an empty 2D array to store the scores
scores = np.zeros((len(trainset) + len(testset), num_classes))

# Switch the model to evaluation mode
model.eval()

# Iterate over the training dataset and compute the scores
counter = 0
with torch.no_grad():
    for i, (images, _) in enumerate(trainloader):
        images = images.to(device)
        outputs = model(images)
        batch_size = outputs.shape[0]
        scores[counter:counter+batch_size] = torch.softmax(outputs, dim=1).cpu().numpy()
        counter += batch_size

# Iterate over the test dataset and compute the scores
with torch.no_grad():
    for i, (images, _) in enumerate(testloader):
        images = images.to(device)
        outputs = model(images)
        batch_size = outputs.shape[0]
        scores[len(trainset) + counter:len(trainset) + counter+batch_size] = torch.softmax(outputs, dim=1).cpu().numpy()
        counter += batch_size

np.save('score_matrix.npy', scores)

scores_train = np.load("score_matrix.npy")

#flip rate corresponds to a noise rate on the two inverted classes

train_val_dataset_noisy_04 = cifar10_noisy(scores=scores_train, flip_rate_fixed=0.4, transform=transform_train, transform_eval = transform_test, target_transform = transform_target)
