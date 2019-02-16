#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:20:08 2018

@author: korhan
"""

import pickle
import matplotlib.pyplot as plt
from os.path import join,dirname,abspath
import numpy as np

rootdir = dirname(abspath(__file__))


def save_obj(obj,name):
    with open(join(rootdir,'obj/'+name+'.pkl'),'wb') as f:
        pickle.dump(obj,f)
        
def load_obj(name):
    with open(join(rootdir,'obj/'+name+'.pkl'),'rb') as f:
        return pickle.load(f)
    
    
def plot_rows(imgs,titles = None):
    """ imgs should be a tuple of the form ((a1,a2,...),(b1,b2,...),(c1,c2,...)) 
        with the same number of images in each
    """
    plt.figure()
    n_col = len(imgs)
    n_row = len(imgs[0])
    for i in range(n_row):
        for j in range(n_col):
            
            ax = plt.subplot(n_row, n_col, n_col*i + j+1)
            ax.axis('off')
            if i==0 and titles != None: ax.set_title(titles[j])
            ax.imshow(imgs[j][i],cmap=plt.cm.gray)
        
    plt.show()    



def computeAccuracy(list1,list2,convert = False): # convert list2 to string list from unicode list
    
    if convert: list2 = [str(list2[l]) for l in range(len(list2)) ]
    
    assert len(list1) == len(list2)
    n_true = float(sum([list1[i] == list2[i] for i in range(len(list1))]))
    
    return n_true / len(list1)



def plot_CMG(rank_acc,N):
    plt.figure()
    plt.plot(range(1,11),rank_acc)
    plt.xlabel('M')
    plt.ylabel('CMG(M,'+ str(N) + ')')
    plt.xticks(range(1,11))
    plt.title('Cumulative Match Characteristic')
    plt.show()

def rankM_acc(ranks):

    rank_acc = np.zeros((10,))
    n = len(ranks)
    
    for M in range(1,11):
        rank_acc[M-1] = (sum(ranks<(M+1))[0]/float(n))
        
    
    return rank_acc
    