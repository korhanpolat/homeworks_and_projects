#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:19:42 2018

@author: korhan
"""
from my_utils import load_obj,save_obj
from os import listdir
from os.path import join,dirname,abspath
import numpy as np
from shutil import copy2
from sklearn.model_selection import train_test_split
import pandas as pd

def copy_files(train_input,source_dir,target_dir):
    """ copy files to new destination """
    for indx,row in train_input.iterrows():
        print indx, row.Image
        if row.Id_Count>7:
            copy2(join(source_dir,row.Image),target_dir)

def new_dir_file_info(new_dir,train_input,save_name=None):
    """ get new dir file info after discarding 
        some images from dataset and copying to new dir 
    """
    onlyfiles = [f for f in listdir(new_dir) if isfile(join(new_dir, f))] # files in new dir
    train_input = train_input[train_input.Image.isin(onlyfiles)] # select entries that exist in new dir
    train_input.drop(columns='Id_Count',inplace=True) # remove old id counts
    ## calculate new image counts
    img_counts = train_input.Id.value_counts().to_dict()
    train_input["Id_Count"] = train_input.Id.apply(lambda x: img_counts[x])
    # drop images with less than 4 samples
    train_input.drop(train_input.index[(train_input.Id_Count < 4)], inplace=True)
    train_input = train_input.reset_index(drop=True)
    
    if save_name is not None: save_obj(train_input,save_name)    
    
    uniq_ids = pd.DataFrame()
    uniq_ids['Id'] = (train_input.Id.unique())
    uniq_ids['Id_count'] = uniq_ids.Id.apply(lambda x: img_counts[x])
    save_obj(uniq_ids,'uniq_ids_8')

    return train_input


def split_train_val_test(train_input,exp_id = 4):
        
    X0 = train_input['Image']
    y0 = train_input['Id']
    
    for split_i in range(1,11):
            
        X_train, X1, y_train, y1 = train_test_split(X0,y0,test_size=0.5, stratify=y0 ,random_state=split_i)
        X_val, X_test, y_val, y_test = train_test_split(X1,y1,test_size=0.5, stratify=y1 ,random_state=split_i)
        
        save_obj(np.array(X_train.index), '/split_ind/train_'+ str(exp_id) + '_' +str(split_i))
        save_obj(np.array(X_val.index), '/split_ind/val_'+ str(exp_id) + '_' +str(split_i))
        save_obj(np.array(X_test.index), '/split_ind/test_'+ str(exp_id) + '_' +str(split_i))
    
def load_train_val_test_idx(split_i=1,exp_id=1):
    
    train_ind = load_obj('/split_ind/train_'+ str(exp_id) + '_' +str(split_i))
    val_ind = load_obj('/split_ind/val_'+ str(exp_id) + '_' +str(split_i))
    test_ind = load_obj('/split_ind/test_'+ str(exp_id) + '_' +str(split_i))

    return train_ind, val_ind,test_ind



#rootdir = '/home/korhan/.kaggle/competitions/whale-categorization-playground'
#trainDir = join(rootdir,'grup6')
#
#np.random.seed(42)
#exp_id=8

#train_input = load_obj('train_grup'+str(exp_id))
#uniq_ids = load_obj('uniq_ids_8')

#img1 = 

#
#for id_name in uniq_ids.Id:
#    same_class_rows = pd.DataFrame(train_input[train_input.Id == id_name])
#
#    same_class_rows=same_class_rows.reset_index(drop=True)
#
#    copy_files(same_class_rows[0:8],join(rootdir,'grup7'),join(rootdir,'grup8'))
#
#
#train_input = new_dir_file_info(join(rootdir,'grup8'),train_input.copy(),save_name='train_grup8')
#

""" split training set """
# split_train_val_test(train_input,exp_id=exp_id)



#train_ind, val_ind,test_ind = load_train_val_test_idx(2,8)

















