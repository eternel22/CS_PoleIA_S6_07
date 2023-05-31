from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--limitData', default=0, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if(args.limitData == 1):
    dataloader.NB_TRAINING_DATA = dataloader.NB_TRAINING_DATA // 5


def training(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    correct = 0
    total = 0
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()
        if(total != 0):
            acc = 100.*correct/total
        else:
            acc = 0

        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Train_Loss: %.4f Train_Accuracy: %.2f%%'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item(), acc))
        sys.stdout.flush()

    history["loss"].append(loss.item())
    history["accuracy"].append(acc)

def test(epoch,net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def valid(epoch,net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
                       
            loss = CEloss(outputs, targets)   
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Epoch #%d\t Validation_Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Validation_Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

    history["val_loss"].append(loss.item())
    history["val_accuracy"].append(acc)


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=2,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net = create_model()
cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr       
    test_loader = loader.run('test')
    valid_loader = loader.run('valid')
    eval_loader = loader.run('eval_train')   
    
    warmup_trainloader = loader.run('warmup')
    training(epoch,net,optimizer,warmup_trainloader)    
   

    valid(epoch, net)  

def afficher():
    import matplotlib.pyplot as plt
    epoch_range = range(1, len(history['loss'])+1)

    plt.figure(figsize=[14,4])
    plt.subplot(1,2,1)
    plt.plot(epoch_range, history['loss'], label='Training')
    plt.plot(epoch_range, history['val_loss'], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epoch_range, history['accuracy'], label='Training')
    plt.plot(epoch_range, history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("results.png")

afficher()
print(history)


test(args.num_epochs+1, net)  

