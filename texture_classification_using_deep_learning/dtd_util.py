#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 00:28:58 2018

@author: korhan
"""

from os.path import join, dirname, basename,exists
from os import makedirs
from json import dump
import pandas as pd
from my_utils import save_obj, load_obj
from shutil import copy2


rootdir = '/home/korhan/Desktop/597/proje/dtd/'


def get_fileList(split_i=1, set_type = 'train'):
    """ set_type : {'train','val','test'} """   
        
    filename = set_type + str(split_i) + '.txt'
    filepath = join(rootdir,'labels',filename)
    with open(filepath) as f:
        file_list = f.readlines()
    f.close()
    file_list = [x.strip() for x in file_list]  # get rid of \n

    return file_list



""" for each split """
def getImgs4split(split_i = 1,set_type = 'train'):
   
    file_list = get_fileList(split_i,set_type)
    
    imgs = [None]*(len(file_list))
    """ for each image """
    for f in file_list:
        
        idx = file_list.index(f)
        imgs[idx] = dict()

        imgs[idx]['label'] = dirname(f)
        imgs[idx]['path'] = join(rootdir,'images',f)
        imgs[idx]['name_id'] = basename(f)[:-4]
        
    with open(join('imgs',set_type + '_imgs_' + str(split_i) + '.txt'), 'w') as outfile:
        dump(imgs, outfile)
        
    return imgs


def get_info(split_i=1,set_type = 'train',save_info=True):
    
    try: info_df = load_obj(set_type+ '_info_'+str(split_i))
    except:    
        file_list = get_fileList(split_i,set_type)
        info_df = pd.DataFrame(columns=['Image_name','Label'])
    
        for f in file_list:
            
            idx = file_list.index(f)
            info_df.loc[idx] = [basename(f) , dirname(f) ]
    
    
        if save_info: save_obj(info_df,set_type+ '_info_'+str(split_i))
    
    return info_df
    


def cropSquare(img):
    
    s1,s2,s3 = img.shape
    min_d=min(s1,s2)

    return img[(s1-min_d)/2:(s1+min_d)/2,(s2-min_d)/2:(s2+min_d)/2]



def copy_files(train_input,source_dir,target_dir):
    """ copy files to new destination """
    for indx,row in train_input.iterrows():
        print indx, row.Image_name
        if not exists(join(target_dir,row.Label)): 
            makedirs(join(target_dir,row.Label))
            
        copy2(join(source_dir,row.Label,row.Image_name),join(target_dir,row.Label))

def create_data_folder(split_i,set_type):
    input_info = get_info(split_i=split_i,set_type=set_type)
    
    source_dir=join(rootdir,'images')
    target_dir=join('/home/korhan/Desktop/597proje/data',set_type)
    if not exists(target_dir): makedirs(target_dir)
    
    copy_files(input_info,source_dir,target_dir)



#split_i = 1
#set_type = 'test'

#create_data_folder(split_i,set_type)



















#imgs = getImgs4split()


#print whosmat(join(rootdir,'imdb','imdb.mat'))
#imdb = loadmat(join(rootdir,'imdb','imdb.mat'))
#
#meta_info = imdb['meta']
#meta_info = meta_info[0,0]
#
