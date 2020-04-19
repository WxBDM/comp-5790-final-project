#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:44:03 2020

@author: Brandon
"""

# General imports
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# package imports
from graph import GraphText
from alexnet import AlexNet
import vgg
import resnet as rn
import densenet as dn

plt.close('all')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# =========================
# == VARIABLES TO CHANGE ==

num_epoches = 3
learning_rate = 0.00001
weight_decay = 0.0005
model = AlexNet() # the model you want to run; see below for model options

size = 224
batch_size = 16
num_workers = 8 # pytorch data loader

train_dir = 'train/'
val_dir = 'val/'

save_csv = True
graph = True

# =========================

# =========================
# model options (a list represents the same model, different options):
# AlexNet:  model = AlexNet()
# VGG:      model = [vgg.vgg11(), vgg.vgg11_bn(), vgg.vgg13(), vgg.vgg13_bn(),
#           vgg.vgg16(), vgg.vgg16_bn(), vgg.vgg19(), vgg.vgg19_bn()]
# ResNet:   model = [rn.resnet18(), rn.resnet34(), rn.resnet50(), rn.resnet101(), 
#               rn.resnet152(), rn.resnext50_32x4d(), rn.resnext101_32x8d(),
#               rn.wide_resnet50_2(), rn.wide_resnet101_2()]
# DenseNet: model = [dn.densenet121(), dn.densenet161(), dn.densenet169(),
#               dn.densenet201()]
# =========================

def graph_training_loss(d):
    '''Graph the training loss for a given network'''
    
    y_axis = d['train_epoch_loss']
    x_axis = range(len(y_axis))
    plt.plot(x_axis, y_axis, color = 'blue',  label = 'Training Loss')
    
    # text-related stuff; see graph.py
    graph = GraphText(plt.gca())
    graph.title("Loss - Training {}".format(d['neuralnet']))
    upper_right_txt = "Learning Rate: {}\nBatch Size: {}".format(d['lr'][0], d['batch_size'][0])
    graph.upperRightBelow(upper_right_txt)
    graph.x_axis_label("epoch")
    graph.y_axis_label("Training - Loss")
    graph.show_legend()
    plt.grid(which = 'both', alpha = 0.4, ls = "--")
    plt.savefig('losses_{}.png'.format(d['neuralnet']), dpi=300, format='png', 
                bbox_inches='tight')

    plt.close('all') # close all graphs; could cause issues.

def graph_accuracy(d):
    '''Graph the accuracy (training and validation) for a given network'''
    
    y_axis_train = d['train_epoch_acc']
    y_axis_val = d['val_epoch_acc']
    x_axis = range(len(y_axis_val))
    
    plt.plot(x_axis, y_axis_train, color = 'blue', label = 'Training Accuracy')
    plt.plot(x_axis, y_axis_val, color = 'red', label = 'Validation Accuracy')
    
    graph = GraphText(plt.gca())
    graph.title("Accuracy: {}".format(d['neuralnet']))
    upper_right_txt = "Learning Rate: {}\nBatch Size: {}".format(d['lr'][0], d['batch_size'][0])
    graph.upperRightBelow(upper_right_txt)
    graph.x_axis_label("epoch")
    graph.y_axis_label("Accuracy")
    graph.show_legend()
    plt.grid(which = 'both', alpha = 0.4, ls = "--")
    plt.ylim(bottom = 0, top = 1)
    plt.savefig('accuracy_{}.png'.format(d['neuralnet']), dpi=300, format='png', 
                bbox_inches='tight')
    
    plt.close('all') # close all graphs; could cause issues.

def save_csv(model, info_d, fname = 'info.csv'):
    '''Save some information of a trained neural network into a CSV file.'''
    
    # reformat the data a bit
    n_datapoints = len(model.train_epoch_loss)
    dict_vals = ['neuralnet', 'lr', 'weight_decay', 'batch_size']
    to_put_back = []
    for val in dict_vals:
        to_put_back.append(d[val])
        d[val] = [d[val]] * n_datapoints
    
    df = pd.DataFrame(data=d)
    df.to_csv("info.csv")

def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=25):
    '''Train the model'''
    
    since = time.time()
    
    train_epoch_loss = []
    train_batch_loss = []
    val_epoch_loss = []
    
    for epoch in range(num_epochs):
    
        # You perform validation test after every epoch
        for phase in ['train', 'val']:
            running_loss = 0
            running_corrects = 0

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
                        
                    if idx % 200 == 0:
                        train_batch_loss.append(loss.item())
                        
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

# save it in a dictionary, recommended to export by having save_csv = True
d = {'neuralnet' : model.name,
     'lr' : learning_rate * n_datapoints,
     'batch_size' : batch_size * n_datapoints,
     'weight_decay' : weight_decay * n_datapoints,
     'train_epoch_loss' : model.train_epoch_loss, 
     'train_epoch_acc' : [model.train_epoch_acc[i].item() for i in 
                                  range(len(model.train_epoch_acc))], 
     'val_epoch_loss' : model.val_epoch_loss, 
     'val_epoch_acc' : [model.val_epoch_acc[i].item() for i in 
                            range(len(model.val_epoch_acc))]}

if save_csv: 
    save_csv(model, d)

if graph:
    graph_training_loss(d)
    graph_accuracy(d)