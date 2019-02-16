#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:58:49 2018

@author: korhan
"""


from os.path import join
#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
#import data_exploration as de
import cv2

from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, transform_matrix_offset_center, img_to_array,array_to_img)


def augmentation_pipeline(img):
    img_arr = img_to_array(img)
    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
#    img_arr = random_greyscale(img_arr, 0.4)

    return img_arr


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
        

#trainDir = '/home/korhan/Desktop/58z/proje/dataset/train';
#
#
#
#img = cv2.imread(join(trainDir,'ff38054f.jpg'))
#
#plt.imshow(img,cmap='gray')

#
#imgs = [
#    random_rotation(img, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
#    for _ in range(5)]
















