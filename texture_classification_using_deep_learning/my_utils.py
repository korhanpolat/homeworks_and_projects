#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:58:48 2018

@author: korhan
"""


import pickle
import pandas as pd
from os.path import join,dirname,abspath

rootdir = dirname(abspath(__file__))



def save_obj(obj,name):
    with open(join(rootdir,'obj',name+'.pkl'),'wb') as f:
        pickle.dump(obj,f,protocol=2)
        
def load_obj(name):
    with open(join(rootdir,'obj',name+'.pkl'),'rb') as f:
        return pickle.load(f)

class MyLogger(object):
    """docstring for ClassName"""

    
    
    def __init__(self,name, base_columns, base_values):	
        super(MyLogger, self).__init__()
        self.base_values = base_values
        self.base_columns = base_columns
    
        try:
            self.log_df = load_obj(name)
        except: 
            self.log_df = pd.DataFrame(columns=self.base_columns)

    def append_row(self,column_names,values):
        
        df = pd.DataFrame([self.base_values+values],columns=self.base_columns+column_names)
        self.log_df = self.log_df.append(df,ignore_index=True,sort=False)

        return self.log_df
    
    def save_df(self):
        save_obj(self.log_df,name)





