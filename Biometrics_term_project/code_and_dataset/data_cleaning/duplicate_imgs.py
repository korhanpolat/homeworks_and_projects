#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:46:05 2018

@author: mostly copied from related kaggle kernel
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import collections
from sklearn.model_selection import StratifiedShuffleSplit
from my_utils import load_obj,save_obj

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

rootdir = '/home/korhan/.kaggle/competitions/whale-categorization-playground'
trainDir = os.path.join(rootdir,'train')
np.random.seed(1)



def plot_images(path, imgs):
    assert(isinstance(imgs, collections.Iterable))
    imgs_list = list(imgs)
    nrows = len(imgs_list)
    if (nrows % 3 != 0):
        nrows = nrows + 2 

    plt.figure(figsize=(24, 6*nrows/3))
    for i, img_file in enumerate(imgs_list):
        with Image.open(os.path.join(path, img_file)) as img:
            ax = plt.subplot(nrows/3, 3, i+1)
            ax.axis('off')
            ax.set_title("#{}: '{}'".format(i+1, img_file))
            ax.imshow(img)
        
    plt.show()

def getImageMetaData(file_path,with_hash=False):
    with Image.open(file_path) as img:
        if with_hash: 
            img_hash = imagehash.phash(img)
            return img.size, img.mode, img_hash
        else:
            return img.size
        
def get_train_input():
    train_input = pd.read_csv(os.path.join(rootdir,'train.csv'))
    
    m = train_input.Image.apply(lambda x: getImageMetaData(trainDir + "/" + x))
    train_input["Hash"] = [str(i[2]) for i in m]
    train_input["Shape"] = [i[0] for i in m]
    train_input["Mode"] = [str(i[1]) for i in m]
    train_input["Length"] = train_input["Shape"].apply(lambda x: x[0]*x[1])
    train_input["Ratio"] = train_input["Shape"].apply(lambda x: x[0]/x[1])
    train_input["New_Whale"] = train_input.Id == "new_whale"
    
    
    img_counts = train_input.Id.value_counts().to_dict()
    train_input["Id_Count"] = train_input.Id.apply(lambda x: img_counts[x])
    return train_input

def is_grey_scale(img_path):

    im = Image.open(img_path).convert('RGB')
    min_len = min(im.size)
    for i,j in np.random.randint(min_len,size=(500,2)):
        r,g,b = im.getpixel((i,j))
        if r != g != b: return False
    return True









""" main """
#
#
#train_input = load_obj('train_info')    
#
## determine duplicate images using the hash
#t = train_input.Hash.value_counts()
#t = t[t > 1]
#duplicates_df = pd.DataFrame(t)
#
#
## get the Ids of the duplicate images
#duplicates_df["Ids"] =list(map(
#            lambda x: set(train_input.Id[train_input.Hash==x].values), 
#            t.index))
#duplicates_df["Ids_count"] = duplicates_df.Ids.apply(lambda x: len(x))
#duplicates_df["Ids_contain_new_whale"] = duplicates_df.Ids.apply(lambda x: "new_whale" in x)
#
#
## Fix error type 1: The same image with the corresponding Id appears multiple time.
#train_input.drop_duplicates(["Hash", "Id"], inplace = True)
#
## Fix error type 2: The same image appears with an Id and as "new_whale".
## => delete the "new_whale" entry
#
## hash cinsinden indexle
#drop_hash = duplicates_df.loc[(duplicates_df.Ids_count>1) & (duplicates_df.Ids_contain_new_whale==True)].index
## hash i drophash icinde olan ve id si new whale olanlari dropla
#train_input.drop(train_input.index[(train_input.Hash.isin(drop_hash) & (train_input.Id=="new_whale"))], inplace=True)
#
## Fix error type 3: The same image appears with different Ids (ambiguous classified).
## => delete all of them
#drop_hash = duplicates_df.loc[(duplicates_df.Ids_count>1) & ((duplicates_df.Ids_count - duplicates_df.Ids_contain_new_whale)>1)].index
#
#train_input.drop(train_input.index[train_input.Hash.isin(drop_hash)], inplace=True)
#
#assert(np.sum(train_input.Hash.value_counts()>1) == 0)
#
#
## drop all rows with new whale id
#train_input.drop(train_input.index[(train_input.Id=="new_whale")], inplace=True)
## drop redundant columns
#train_input.drop(columns=['New_Whale', 'Id_Count'],inplace=True)
## calculate new image counts
#img_counts = train_input.Id.value_counts().to_dict()
#train_input["Id_Count"] = train_input.Id.apply(lambda x: img_counts[x])
## drop images with less than 4 samples
#train_input.drop(train_input.index[(train_input.Id_Count < 4)], inplace=True)
#
#train_input['is_grey'] = train_input.Image.apply(lambda x: is_grey_scale(os.path.join(trainDir,x)))
#
#
#
#train_input = train_input.reset_index(drop=True)
#save_obj(train_input,'train_info')
#

#
#
#
#
#
#
#
#
#
#
#

