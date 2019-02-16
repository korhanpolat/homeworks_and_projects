#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 03:01:28 2018

@author: korhan


"""
from db_utils import load_train_val_test_idx,load_obj,save_obj
from os.path import join,dirname,abspath
from eigen_fluke import eigen_fluke_transformer,symmetry_correction,read_img,project_to_n_pc
from sklearn import svm
import numpy as np
import pandas as pd

""" parameters """
rootdir = rootdir = dirname(abspath(__file__))
trainDir = join(rootdir,'images')
exp_id=8
method='svm'
split_n = 1



train_input = load_obj('train_info'+str(exp_id))


def get_features_for_images(indices,W):

    X = np.zeros((len(indices),len(W)))
    y = list()

    for i in range(len(indices)):

        img = read_img(join(trainDir,train_input.Image[indices[i]]),gray=True)

        X[i] = project_to_n_pc(img,W)
        y.append(train_input.Id[indices[i]])

    return X,y

def fit_SVM(train_data, train_labels,save_title=None ): 

    clf = svm.LinearSVC()
    clf.fit(train_data, train_labels)
    print 'SVM fitted'
    if save_title is not None: save_obj(clf,save_title)
    return clf

def knearest_neighbour(train_X,train_y,train_ind, x_test, k):

    # create list for distances and targets
    distances = []
    
    train_inp = train_input.loc[train_ind]
    n_feature_per_class = train_inp.Id.value_counts().to_dict()


    for i in range(len(train_X)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - train_X[i, :])))
        # add it to list of distances
        distances.append([distance, train_y[i]])

    # sort the list
    distances = sorted(distances)

    df = pd.DataFrame(distances[0:k],columns=['Dist','Id'])
    nn_per_class = df.Id.value_counts().to_dict()
    
    preds = dict.fromkeys(df.Id.unique(),0)
        
    for row in df.itertuples():
#        print row.Id, row.Dist
        preds[row.Id] += row.Dist/(nn_per_class[row.Id]*n_feature_per_class[row.Id])
        
    likelihoods = pd.DataFrame.from_dict(preds,orient='index')
    likelihoods.reset_index(inplace=True)

    likelihoods.columns = ['Id','dist']
    
    likelihoods = likelihoods.sort_values('dist')
    likelihoods.reset_index(drop=True,inplace=True)

    likelihoods['Prob'] = likelihoods.dist.apply(lambda x: 1000/x)


#    return likelihoods.iloc[0:1]    
    return likelihoods.Id.iloc[0]
    
def knn_classify(train_X,train_y,train_ind,X,k):
    
    pred_labels = []
    
    for i in range(len(X)):
        x_test = X[i]
        pred_labels.append(knearest_neighbour(train_X,train_y,train_ind,x_test,k))
    
    return pred_labels
    


def computeAccuracy(list1,list2,convert = False): # convert list2 to string list from unicode list
    
    if convert: list2 = [str(list2[l]) for l in range(len(list2)) ]
    
    assert len(list1) == len(list2)
    n_true = float(sum([list1[i] == list2[i] for i in range(len(list1))]))
    
    return n_true / len(list1)




class PCA_Classification(object):
        
    def __init__(self, split_i,exp_id,n_components, save_pca=False):
        """ PCA is computed at initialization once """
 
        self.split_i = split_i
        self.exp_id = exp_id
        self.train_ind , self.val_ind, self.test_ind = load_train_val_test_idx(split_i,exp_id)

        try: 
            self.main_W = load_obj('pca_obj/pca_weights' + '_' +str(self.split_i)+'_'+str(self.exp_id))
            print 'PCA loaded'
        except:
            self.main_W = eigen_fluke_transformer(train_input.loc[self.train_ind],n_components)
            if save_pca: save_obj(self.main_W,'pca_obj/pca_weights' + '_' +str(self.split_i)+'_'+str(self.exp_id))



    def do_training(self, dist_thresh,first_n_comp=None,method='svm'):

        self.W = symmetry_correction(self.main_W,dist_thresh=dist_thresh)
        self.method = method
        
        if first_n_comp is not None: self.first_n_comp = first_n_comp
        else: self.first_n_comp = len(self.W)
            
        X,y = get_features_for_images(self.train_ind,self.W[:first_n_comp])
        print('Training features are ready!')

        if method=='svm': self.svm = fit_SVM(X,y)
        elif method=='knn': 
            self.train_X = X 
            self.train_y = y
    
#        return classifier
    
    
    def do_testing(self,indices):
        
#        W = load_obj('pca_weights'+str(self.split_i))
        
        X,y = get_features_for_images(indices,self.W[:self.first_n_comp])
        
        if method=='svm': 
            self.pred_labels = self.svm.predict(X)    

        elif method=='knn': 
            self.pred_labels = knn_classify(self.train_X,self.train_y,self.train_ind,X,k=self.k)
            
        acc = computeAccuracy(y,self.pred_labels,convert=True)
        
        print('Accuracy for first ' + str(self.first_n_comp) + ' components is ' + str(acc))
        
        return acc
    

""" main """

col_list = ['Split','Set_type','Dist_thresh','n_comp','Accuracy','method']


try: pca_accuracies = load_obj('pca_acc_'+str(exp_id))
except: pca_accuracies = pd.DataFrame(columns=col_list)

dist_thresh = 1.03
for split_i in range(1,split_n+1): #=1
    print 'Split:', split_i

    classifier = PCA_Classification(split_i,exp_id=exp_id, n_components=1000)

    classifier.do_training(dist_thresh=dist_thresh,method=method)

#        # train accuracy
#        acc = classifier.do_testing(classifier.train_ind)
#        df = pd.DataFrame([[split_i,'train',dist_thresh,classifier.first_n_comp,acc,method]], columns=col_list)
#        pca_accuracies = pca_accuracies.append(df,ignore_index=True)

#        # validation accuracy
    acc = classifier.do_testing(classifier.test_ind)
    df = pd.DataFrame([[split_i,'test',dist_thresh,classifier.first_n_comp,acc,method]], columns=col_list)

    pca_accuracies = pca_accuracies.append(df,ignore_index=True)

    save_obj(pca_accuracies,'pca_acc_'+str(exp_id))

print 'Mean accuracy for experiment ' +str(exp_id)+' and ' +str(split_n)+' splits is:' ,pca_accuracies.Accuracy[(pca_accuracies.Dist_thresh==dist_thresh ) & (pca_accuracies.method==method)].mean()

    
    









    
    














