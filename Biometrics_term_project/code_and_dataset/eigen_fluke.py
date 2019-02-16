#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:40:37 2018

@author: korhan
"""

#from db_utils import load_train_val_test_idx
import numpy as np
from my_utils import load_obj,save_obj,plot_rows
import cv2
from opencv_fncs import read_img
from os.path import join,dirname,abspath
from sklearn.decomposition import PCA
#from matplotlib import pyplot as plt

rootdir = dirname(abspath(__file__))
trainDir = join(rootdir,'images')

#train_input = load_obj('train_grup6')

ratio = 2.5    
h=100    
w = int(h * ratio)   


def reshape_img(img,w=250,h = 100):
    
    img = cv2.resize(img,(w,h))        
    # vectorize
    img_vec = img.reshape(h*w)
    
    return img_vec

def eigen_fluke_transformer(train_rows,n_components = 200):
    
    
    X = np.empty((len(train_rows),h*w))    

    i = 0
    for row in train_rows.itertuples():
        # read and resize
        img = read_img(join(trainDir,row.Image),gray=True)
        
        X[i] = reshape_img(img,w=w,h = h)
        i += 1
        
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X)
    print('PCA fitted')

#    eigenflukes = pca.components_.reshape((n_components, h, w))
    
    return pca.components_ 


def project_to_n_pc(img,W):
    
    img_vec = reshape_img(img)
    # compess and decompress
    compressed = np.dot(W,img_vec)
    
    return compressed

def eigen_fluke_recons(img,W,n):
    
    
    compressed = project_to_n_pc(img,W[:n])
    
    recons = np.dot(W[:n].T,compressed)
    # reshape 
    img_recons = (recons.reshape(h,w))
    img_recons = np.uint8(255*(img_recons - np.min(img_recons))/(np.max(img_recons) - np.min(img_recons)))
    # resize to original dim
    img_recons = cv2.resize(img_recons,(img.shape[1],img.shape[0]))
    
    return img_recons

def symmetry_correction(W,dist_thresh=.9):
    
    passed_eigens = list()
    
    
    for i in range(len(W)):
        W_img = W[i].reshape(100,250)
        
        left_half = W_img[:,:125]
        right_half = np.flip(W_img[:,125:],1)
    
        dist = np.linalg.norm(np.abs(right_half-left_half))
        if dist < dist_thresh: passed_eigens.append(i)
    
    W_passed = W[passed_eigens]
#    save_obj(W_passed,'pca_weights_2')    
    return W_passed    


""" main """

#train_ind , val_ind,test_ind = load_train_val_test_idx(split_i=1,exp_id=8)

#W = eigen_fluke_transformer(train_ind,n_components = 400,n_img = 500)
#W = load_obj('pca_obj/pca_weights'+str(1))
#for i in range(1,5): # i=64
#
#    img = read_img(join(trainDir,train_input.Image[train_ind[i]]),gray=True)
#    img_recons = eigen_fluke_recons(img,W,10)
#    #img_binary = cv2.Canny(img_recons,100,200)
#    
#    
#    
#    img_binary = threshold_mask(img_recons)
#    
#    plot_rows(((img,),(img_recons,),(img_binary,)))
#    















