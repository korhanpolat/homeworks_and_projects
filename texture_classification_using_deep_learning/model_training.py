#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 22:46:28 2018

@author: korhan
"""

""" training functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from torchvision import transforms,datasets
import copy
from os.path import join,dirname,abspath
from model_loading import GramClassifier,GramSum5,GramConv5
from my_utils import MyLogger


def train_model(model, criterion, optimizer, scheduler, num_epochs=25,Logger=None,pretrain=True,previous_epochs=0):
    since = time.time()

    best_acc = 0.0
    phase_list = ['train', 'val'] 
        
    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in phase_list:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for itr, data_inp_lab in enumerate(dataloaders[phase]):
                inputs, labels = data_inp_lab

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if itr%50 ==  0: print(itr,running_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Duration: {:.2f}'.format(
                phase, epoch_loss, epoch_acc, time.time()-epoch_start))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights    
    model.load_state_dict(best_model_wts)


    return model,state


if __name__ == '__main__':   

    """ parameters """

    for split_i in range(1,5):

        rootdir = dirname(abspath(__file__))
        data_dir = join(rootdir,'kth','sp'+str(split_i))   
        n_class = 11

        model_name = 'gram_sum_5'
        exp_name = 'kth_gram_sum_5'

        epoch_train = 5
        criterion = nn.CrossEntropyLoss()
            
        """ data loading """
        batch_size = 32

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=4) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        """ =========== 
        load the model 
        =========== """


        if 'gram_conv' in model_name : model_ft = GramConv5(n_class=n_class,d_conv=d_conv)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)
        # set parameters
        Learning_rate = 0.002
        momentum = .9
        # optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad,model_ft.parameters()),lr=Learning_rate,momentum=momentum)
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad,model_ft.parameters()))
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.7)

        model_ft,state = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=epoch_train,Logger=Logger,pretrain=False)

        torch.save(model_ft.state_dict(),join(dirname(abspath(__file__)),'saved_models',model_name+'_'+str(split_i)))


