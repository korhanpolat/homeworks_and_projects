#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:25:25 2018

@author: korhan
"""

import torch.nn as nn
import torchvision
import numpy as np
import torch
from os.path import join,dirname,abspath

class GramMatrix(nn.Module):
    """ compute correlations of a tensor"""
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G
        
        
class VectorizeTriu(nn.Module):
    """ vectorize upper triangle of an input matrix """
    def forward(self, x):
        row_idx, col_idx = np.triu_indices(x.shape[2])
        row_idx = torch.LongTensor(row_idx).cuda()
        col_idx = torch.LongTensor(col_idx).cuda()
        x = x[:, row_idx, col_idx]
        return x        
        
        
class Vgg_FeatExtr(nn.Module):
    """ extract features for Conv5-3 layer """
    def __init__(self,freeze=True):
        super(Vgg_FeatExtr,self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        # freeze parameters not to train        
        if freeze: 
            for i, param in vgg_model.named_parameters(): 
                param.requires_grad = False

        self.Conv5 = nn.Sequential(
            *list(vgg_model.features.children())[:34])

    def forward(self,x):
        x = self.Conv5(x)
        return x


class FC_layer(nn.Module):
    """ classifier that is used as a template for all my models """
    def __init__(self,d_in,d_out,h1=4096,h2=2048,dropout_p=0.3):
        super(FC_layer,self).__init__()
        
        fc_features = [nn.Linear(in_features= d_in, out_features=h1, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=dropout_p),
                       nn.Linear(in_features=h1, out_features=h2, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=dropout_p),
                       nn.Linear(in_features=h2, out_features=d_out, bias=True)]

        self.fc = nn.Sequential(*fc_features)
        
    def forward(self,x):
        pred = self.fc(x)        
        return pred


class GramClassifier(nn.Module):
    """ model that is used for experiment #1 """
    def __init__(self,d_ch,n_class,freeze=True):
        super(GramClassifier,self).__init__()
        self.d_ch = d_ch
        self.vgg_feat = Vgg_FeatExtr(freeze=freeze)
        
        self.conv_channels = nn.Conv2d(512, d_ch, 1)
        
        self.gram = GramMatrix()
        self.vectorize = VectorizeTriu()
        
        self.fc = FC_layer(d_in=d_ch*(d_ch-1)/2+d_ch,d_out=n_class,h1=4096,h2=2048)

    def load_trained_conv(self,name):
        self.conv_channels.load_state_dict(torch.load(join(dirname(abspath(__file__)),'saved_models',name)))

    def forward(self,x):
        x = self.vgg_feat(x)
        x = self.conv_channels(x)        
        self.x = self.gram.forward(x)        
        x = self.vectorize(self.x)
        pred = self.fc(x)
        
        return pred

    
class GramSum5(nn.Module):
    """ model that is used for experiment #2 """
    def __init__(self,n_class,freeze=True):
        super(GramSum5,self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)        
        if freeze: 
            for i, param in vgg_model.named_parameters(): param.requires_grad = False
        # extract features from 5 different layers
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:18])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[18:27])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[27:34])

        self.gram = GramMatrix()
        
        self.fc = FC_layer(d_in=512+512+256+128+64,d_out=n_class,h1=4096,h2=2048,dropout_p=0.5)
        
    def forward(self,x):
        # extract features
        g1 = self.Conv1(x)
        g2 = self.Conv2(g1)
        g3 = self.Conv3(g2)
        g4 = self.Conv4(g3)
        g5 = self.Conv5(g4)        
        # compute correlations
        g1 = self.gram.forward(g1) 
        g2 = self.gram.forward(g2) 
        g3 = self.gram.forward(g3) 
        g4 = self.gram.forward(g4) 
        g5 = self.gram.forward(g5) 
        # sum along axis        
        g1 = torch.sum(g1,dim=1)
        g2 = torch.sum(g2,dim=1)
        g3 = torch.sum(g3,dim=1)
        g4 = torch.sum(g4,dim=1)
        g5 = torch.sum(g5,dim=1)
        # concatenate
        out = torch.cat((g1,g2,g3,g4,g5),1)
        
        pred = self.fc(out)        
        return pred


class GramConv5(nn.Module):
    """ model that is used for experiment #3 """
    def __init__(self,n_class,d_conv=1,freeze=True):
        super(GramConv5,self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)        
        if freeze: 
            for i, param in vgg_model.named_parameters(): param.requires_grad = False

        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:18])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[18:27])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[27:34])

        self.gram = GramMatrix()
        # define dx1 convolutions
        self.gramconv1 = nn.Conv2d(1,d_conv,kernel_size=(64,1))
        self.gramconv2 = nn.Conv2d(1,d_conv,kernel_size=(128,1))
        self.gramconv3 = nn.Conv2d(1,d_conv,kernel_size=(256,1))
        self.gramconv4 = nn.Conv2d(1,d_conv,kernel_size=(512,1))
        self.gramconv5 = nn.Conv2d(1,d_conv,kernel_size=(512,1))
        
        self.relu = nn.ReLU()

        self.fc = FC_layer(d_in=d_conv*(512+512+256+128+64),d_out=n_class,h1=4096,h2=2048,dropout_p=0.3)
        
        self.d_conv = d_conv

    def forward(self,x):

        g1 = self.Conv1(x)
        g2 = self.Conv2(g1)
        g3 = self.Conv3(g2)
        g4 = self.Conv4(g3)
        g5 = self.Conv5(g4)        
        
        g1 = self.gram.forward(g1) 
        g2 = self.gram.forward(g2) 
        g3 = self.gram.forward(g3) 
        g4 = self.gram.forward(g4) 
        g5 = self.gram.forward(g5) 
        # convolve Gram matrices                
        g1 = self.relu(self.gramconv1(g1.unsqueeze_(1)).view(-1,self.d_conv*64))
        g2 = self.relu(self.gramconv2(g2.unsqueeze_(1)).view(-1,self.d_conv*128))
        g3 = self.relu(self.gramconv3(g3.unsqueeze_(1)).view(-1,self.d_conv*256))
        g4 = self.relu(self.gramconv4(g4.unsqueeze_(1)).view(-1,self.d_conv*512))
        g5 = self.relu(self.gramconv4(g5.unsqueeze_(1)).view(-1,self.d_conv*512))
        out = torch.cat((g1,g2,g3,g4,g5),1)
        pred = self.fc(out)
        return pred

