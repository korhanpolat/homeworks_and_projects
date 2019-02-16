#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:52:34 2018

@author: korhan


this script performs SIFT feature matching to make predictions.  

you should change exp_id parameter from 4 to 8 to switch between
two different experiment sets

for each split, first part is training, and SIFT features are pooled
to training features list together with corresponding classes. the 
second part is testing and for each test image, feature matching to 
training features is performed and the class with highest number of matches 
is chosen.

algorithm parameters can be changed from "parameters" section

"""

from opencv_fncs import read_img,gaus_blur,resize_img, first_n_good_matches, feat_detector
from os.path import join,dirname,abspath
from db_utils import load_train_val_test_idx,save_obj
import numpy as np
import pandas as pd
from my_utils import computeAccuracy,load_obj,rankM_acc,plot_CMG
import time

""" parameters """

rootdir = dirname(abspath(__file__))
trainDir = join(rootdir,'images')
exp_id=8

method = 'sift' #'orb'
resized_pix = 1.5e5
threshold = 120
max_match = 40
split_n = 1

""" initializations """
train_input = load_obj('train_info'+str(exp_id))
uniq_ids = load_obj('uniq_ids_'+str(exp_id))

col_list = ['method', 'resized_pix', 'threshold', 'max_match', 'split_i', 'acc']

accuracies_list = []

#
try: acc_df=load_obj('feat_det_acc_'+str(exp_id))
except: acc_df = pd.DataFrame(columns=col_list)

mean_acc = 0
rank_acc = np.zeros((10,))

for split_i in range(1,split_n+1):
    t = time.time()

    train_ind, val_ind,test_ind = load_train_val_test_idx(split_i=split_i,exp_id=exp_id)
    train_inp = train_input.loc[train_ind]
    
    trainig_features = list()
    n_feature_per_class = dict()
    
    
    for id_name in uniq_ids.Id:
    #for id_name in ['w_73d5489']:
        print 'ID:', id_name
        
        # %% in-class feature extraction
        
        same_class_rows = train_inp[train_inp.Id == id_name]
        same_class_rows = same_class_rows.reset_index(drop=True)
        
        same_class_features = [set() for i in range(len(same_class_rows))]
        class_kp =  [[] for i in range(len(same_class_rows))]
        class_desc =  [[] for i in range(len(same_class_rows))]
        
        # find descriptors for each image for the class
        for i in range(len(same_class_rows)):
            img = gaus_blur(resize_img(read_img(join(trainDir,same_class_rows.Image[i]),gray=True),resized_pix))
            class_kp[i],class_desc[i] =  feat_detector(img,method=method)
        
        
        for i in range(len(same_class_rows)):
        
            kp1,desc1 = class_kp[i],class_desc[i]
        
            for j in range(i+1,len(same_class_rows)):
                
#                                print i,j
                
                kp2,desc2 = class_kp[j],class_desc[j]
        
                matches = first_n_good_matches(kp1,desc1 ,kp2,desc2 ,n=max_match,threshold = threshold)
                
                [same_class_features[i].add(m.queryIdx) for m in matches]
                [same_class_features[j].add(m.trainIdx) for m in matches]
                
        
        same_class_features  = [sorted(list(same_class_features[i])) for i in range(len(same_class_features ))]
        
        n_featrue = 0
        for i in range(len(same_class_rows)):
            temp = [(class_kp[i][j],class_desc[i][j],id_name) for j in same_class_features[i]]
            trainig_features.extend(temp)
            n_featrue += len(temp)
    
        n_feature_per_class[id_name] =  n_featrue
    
    # %% validation classification
    print '_____Validation phase_____'

    val_inp = train_input.loc[test_ind]
    val_inp = val_inp.reset_index(drop=True)
    
    true_labels = []
    pred_labels = []
    ranks = np.zeros((len(val_inp),1))
    

    for i in range(len(val_inp)):  #i = 0
#                    for i in [1]:    
        print 'test img #', i+1

        img = gaus_blur(resize_img(read_img(join(trainDir,val_inp.Image[i]),gray=True),resized_pix))
        val_kp,val_desc =  feat_detector(img,method=method)
        
        train_kp=[l[0] for l in trainig_features]
        train_desc =np.array([l[1] for l in trainig_features])
        # match test img features with pool of training features 
        matches = first_n_good_matches(val_kp,val_desc,train_kp,train_desc,n=100,threshold=threshold,homography_test=False)
        
        preds = pd.Series([trainig_features[m.trainIdx][2] for m in matches])
        likelihoods = pd.DataFrame( preds.value_counts())
        likelihoods.reset_index(inplace=True)
        likelihoods.columns = ['Id','Match_count']
        
        likelihoods['Prob'] =[float(row.Match_count)/n_feature_per_class[row.Id] for row in likelihoods.itertuples()]
        likelihoods.sort_values('Prob',ascending=False,inplace=True)
        likelihoods.reset_index(inplace=True,drop=True)
#                        print 'len(likelihood): ', len(likelihoods)
        
        pred_labels.append( likelihoods.Id[likelihoods.Prob==likelihoods.Prob.max()].values[0])
        true_labels.append( val_inp.Id[i] )
        
        # find true label's rank
        try: rank = likelihoods[likelihoods.Id==val_inp.Id[i]].index[0] + 1
        except: rank = len(uniq_ids)
        ranks[i] = rank
        
    rank_acc += rankM_acc(ranks)
    
    acc=computeAccuracy(pred_labels,true_labels)

    mean_acc += acc

    df = pd.DataFrame([[method,resized_pix,threshold,max_match,split_i,acc]],
                      columns=col_list)
    
    acc_df = acc_df.append(df,ignore_index=True)

    print df
    print 'Training and testing complete:',time.time()-t , 'seconds'                    

    save_obj(acc_df,'feat_det_acc_'+str(exp_id))
#                    time.sleep(30)

print 'Mean accuracy for experiment ' +str(exp_id)+ ' and ' + str(split_n) + ' splits is ' + str(mean_acc/split_n)
    

rank_acc = rank_acc/split_n
plot_CMG(rank_acc,len(uniq_ids))













