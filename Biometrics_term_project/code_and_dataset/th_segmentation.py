#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 18:23:25 2018
@author: korhan

this script is useless. 

I tried to segment flukes by thresholding 
methods but it did not work out well. 


"""

from my_utils import plot_rows,load_obj
from os.path import join,dirname,abspath
import numpy as np
from opencv_fncs import read_img, gaus_blur, try_thresholdings
#from matplotlib import pyplot as plt
from skimage.filters import threshold_minimum,threshold_otsu
#from scipy.ndimage import label, generate_binary_structure
import cv2
from skimage.morphology import selem,binary_opening,dilation

rootdir = dirname(abspath(__file__))
trainDir = join(rootdir,'images')

np.random.seed(1)

train_input = load_obj('train_info4')
uniq_ids = load_obj('uniq_ids_4')

def inv_bool(img):
    """ img : bool array """
    
    h,w = img.shape

    regions = ((0 , h/6 , w/2-h/6, w/2+h/6),(5*h/6 ,h ,0 ,h/6 ),(5*h/6 ,h ,w-h/6,w ))
    n_pixel = 0
    n_true = 0
    
    for r in regions:        

        n_pixel += (r[1]-r[0]) * (r[3]-r[2])
        n_true += np.sum(img[r[0]:r[1],r[2]:r[3]])
    
    if n_true > n_pixel/2: img = ~img
    return img

def combine_binary(bin1,bin2):
    
    temp = bin1 & bin2

    n_pixel = temp.shape[0] * temp.shape[1]
    
    if np.sum(temp) > 0.1*n_pixel: return temp
    else: return bin1 | bin2


def threshold_mask(img,blur=True,plot_result=False):
    
    
    if blur: img = gaus_blur(img)    
    
    thresh_otsu = threshold_otsu(img)
    binary_otsu = (img > thresh_otsu)
    binary_otsu = inv_bool(binary_otsu)
    
    try:
        thresh_min = threshold_minimum(img)
        binary_min = (img > thresh_min)
        binary_min = inv_bool(binary_min)
        binary_comb = combine_binary(binary_min , binary_otsu)
    except:
        binary_comb = binary_otsu
    
    bin_open = binary_opening(binary_comb,selem.disk(4))
    bin_o_d =  dilation(bin_open,selem.disk(15))
    
    img_masked = np.uint8(cv2.blur(np.float64(bin_o_d),(9,9))*img)

#    if save_path is not None: 
#        cv2.imwrite(join(rootdir,save_path,img_name[:-4] + '.png'),np.uint8(bin_o_d*255) )
#    
    if plot_result: 
        plot_rows(((img,),(binary_otsu,),(bin_o_d,)),('original','otsu','open & dilate'))
        
    return bin_o_d




""" main """

#try_thresholdings(read_img(join(trainDir,train_input.Image[153]),gray=True))


for i in range(110,120):
    print i
    img = read_img(join(trainDir,train_input.Image[i]),gray=True)
    
    bin_o_d = threshold_mask(img,plot_result=True)
    
    
    
    
    


