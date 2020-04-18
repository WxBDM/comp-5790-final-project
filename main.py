#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:44:03 2020

@author: Brandon
"""

# Initial setup - lots of imports.

import matplotlib.pyplot as plt
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from alexnet import AlexNet
import vgg
import resnet as rn

# ===== ONE TIME RUN =====
# run the below code in the interpreter before starting.

#source_dir = 'train/'
#dest_dir = 'val/'
#dirs = os.listdir(source_dir)
#dirs.sort()
#
#for dir in dirs:
#    if not os.path.isdir(dest_dir + dir):
#        os.makedirs(dest_dir + dir)
#        path_source = source_dir + dir
#        path_dest = dest_dir + dir
#        images = os.listdir(path_source)
#        
#        for i in range(10):
#            shutil.copy(path_source + '/' + images[i], path_dest)

# =========================


# =========================
# == VARIABLES TO CHANGE ==

num_epoches = 10
learning_rate = 0.0001
weight_decay = 0.0005
model = vgg.vgg13_bn() # the model you want to run; see below for model options

size = 224
batch_size = 32
num_workers = 8 # pytorch data loader

train_dir = 'train/'
val_dir = 'val/'

save_csv = True
show_graphs = True

# =========================

# =========================
# model options (a list represents the same model, different options):
# AlexNet:  model = AlexNet()
# VGG:      model = [vgg.vgg11(), vgg.vgg11_bn(), vgg.vgg13(), vgg.vgg13_bn(),
#           vgg.vgg16(), vgg.vgg16_bn(), vgg.vgg19(), vgg.vgg19_bn()]
# ResNet:   model = [rn.resnet18, rn.resnet34, rn.resnet50, rn.resnet101, 
#               rn.resnet152, rn.resnext50_32x4d, rn.resnext101_32x8d,
#               rn.wide_resnet50_2, rn.wide_resnet101_2]
# =========================



def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    train_epoch_loss = []
    val_epoch_loss = []
    
    for epoch in range(num_epochs):
        running_loss = 0
        running_corrects = 0
        
        # You perform validation test after every epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            for idx, (inputs, labels) in enumerate(data_loader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero accumulated gradients
                optimizer.zero_grad()
                
                # During train phase we want to remember history for grads
                # and during val we do not want history of grads
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                        
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loader[phase].dataset)
            
            print('Epoch {}/{} - {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, 
                  num_epochs, phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                val_epoch_loss.append((epoch_loss, epoch_acc))
                model.val_epoch_loss.append(epoch_loss)
                model.val_epoch_acc.append(epoch_acc)
                scheduler.step(loss.item())
            else:
                train_epoch_loss.append((epoch_loss, epoch_acc))
                model.train_epoch_loss.append(epoch_loss)
                model.train_epoch_acc.append(epoch_acc)
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                  time_elapsed % 60))
    
    return model

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'val': ImageFolder(val_dir, transform=data_transforms['val']),
}

data_loader = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers) for x in ['train', 'val']
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

model = model.to(device)
model = train(model, data_loader, criterion, optimizer, scheduler, num_epochs = num_epoches)

n_datapoints = len(model.train_epoch_loss)

if save_csv:
    # put values into csv file, manually add in learning rate and weight decay.
    d = {'neuralnet' : [model.name] * n_datapoints,
         'lr' : [learning_rate] * n_datapoints,
         'batch_size' : [batch_size] * n_datapoints,
         'weight_decay' : [weight_decay] * n_datapoints,
         'train_epoch_loss' : model.train_epoch_loss, 
         'train_epoch_acc' : [model.train_epoch_acc[i].item() for i in 
                                  range(len(model.train_epoch_acc))], 
         'val_epoch_loss' : model.val_epoch_loss, 
         'val_epoch_acc' : [model.val_epoch_acc[i].item() for i in 
                                range(len(model.val_epoch_acc))]}
    
    df = pd.DataFrame(data=d)
    df.to_csv("info.csv")

if show_graphs:
    
    # Loss function: training
    plt.plot(d['train_epoch_loss'], list(range(n_datapoints)))
    plt.title("Loss function - Training {}".format(model.name))
    plt.ylabel()
    plt.show()
    pass