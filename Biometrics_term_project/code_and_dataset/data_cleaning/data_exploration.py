#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:21:16 2018

@author: korhan
"""

import os
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import cv2
from my_utils import load_obj

rootdir = '/home/korhan/.kaggle/competitions/whale-categorization-playground'
trainDir = os.path.join(rootdir,'grup6')

def plot_images_for_filenames(filenames, labels=None, rows=4):
    imgs = [plt.imread(os.path.join(trainDir,filename)) for filename in filenames]
    
    return plot_images(imgs, labels, rows)
    
        
def plot_images(imgs, labels=None, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
#        cv2.imshow('label',imgs[i])
        plt.imshow(imgs[i],cmap='gray')




exp_id=4

train_input = load_obj('train_grup'+str(exp_id))
uniq_ids = load_obj('uniq_ids_'+str(exp_id))

i = 5

#id_name = uniq_ids.Id.loc[i]
id_name='w_3b0894d'

filenames = list(train_input.Image[train_input.Id==id_name])

plot_images_for_filenames(filenames,rows=3)







































