# -*- coding: utf-8 -*-
"""
@author: cmoralesb
@author: csanmiguel
"""
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

class CustomGeneratorImg(Sequence):
    
    '''A custom generator for feed batchs into the N, avoiding Out of Memory errors
       when training with GPU (large dataset).It has to be indicated as partof the
       fit_on_generator method for keras models '''
    
    def __init__(self, X, Y, batch_size):
        
        self.data_X = X
        self.data_Y = Y
        self.batch_size = batch_size
        
    def __len__(self):
        
        #Number of batches that this generator is meant to produce
        return (np.ceil(len(self.data_Y) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        
        #Given the batch number index (idx) within the Sequence, retunr a list 
        # that consists of data batch and the ground-truth (GT)
        #batch_X = self.data_X[idx * self.batch_size : (idx+1) * self.batch_size]
        #batch_Y = self.data_Y[idx * self.batch_size : (idx+1) * self.batch_size]
        
        #But we want to make sure in every batch are the same numbers of classes:
        batch_X = []
        batch_Y = []
        
        nclasses = len(np.unique(self.data_Y))
        k = int(np.ceil(self.batch_size/nclasses))
        for i in range(nclasses):
            lbl_index = np.where(self.data_Y == np.unique(self.data_Y)[i])
            k_index = np.random.choice(lbl_index[0], replace=False, size = k)
            for j in range(k):
                batch_X.append(self.data_X[k_index[j]])
                batch_Y.append(self.data_Y[k_index[j]])
        
        #To arrays & shuffle
        batch_X = np.squeeze(batch_X)
        batch_Y = np.squeeze(batch_Y)
        batch_X = batch_X.reshape((batch_X.shape[0], batch_X.shape[1],
                                   batch_X.shape[2], batch_X.shape[3], 1))
        
        batch_X, batch_Y = shuffle(batch_X, batch_Y)
        
        #Cut again to the batch size
        batch_X = batch_X[0:self.batch_size]
        batch_Y = batch_Y[0:self.batch_size]
        
        return np.array(batch_X), np.array(batch_Y)
    
class CustomGenerator(Sequence):
    
    '''A custom generator for feed batchs into the N, avoiding Out of Memory errors
       when training with GPU (large dataset).It has to be indicated as partof the
       fit_on_generator method for keras models '''
    
    def __init__(self, Ximg, Xc, Y, batch_size):
        
        self.data_Ximg = Ximg
        self.data_Xc = Xc
        self.data_Y = Y
        self.batch_size = batch_size
        
    def __len__(self):
        
        #Number of batches that this generator is meant to produce
        return (np.ceil(len(self.data_Y) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        
        #Given the batch number index (idx) within the Sequence, retunr a list 
        # that consists of data batch and the ground-truth (GT)
        #batch_X = self.data_X[idx * self.batch_size : (idx+1) * self.batch_size]
        #batch_Y = self.data_Y[idx * self.batch_size : (idx+1) * self.batch_size]
        
        #But we want to make sure in every batch are the same numbers of classes:
        batch_Ximg = []
        batch_Xc = []
        batch_Y = []
        
        nclasses = len(np.unique(self.data_Y))
        k = int(np.ceil(self.batch_size/nclasses))
        for i in range(nclasses):
            lbl_index = np.where(self.data_Y == np.unique(self.data_Y)[i])
            k_index = np.random.choice(lbl_index[0], replace=False, size = k)
            for j in range(k):
                batch_Ximg.append(self.data_Ximg[k_index[j]])
                batch_Xc.append(self.data_Xc[k_index[j]])
                batch_Y.append(self.data_Y[k_index[j]])
        
        #To arrays & shuffle
        batch_Ximg = np.squeeze(batch_Ximg)
        batch_Xc = np.squeeze(batch_Xc)
        batch_Y = np.squeeze(batch_Y)
        batch_Ximg = batch_Ximg.reshape((batch_Ximg.shape[0], batch_Ximg.shape[1],
                                         batch_Ximg.shape[2], batch_Ximg.shape[3], 1))
        batch_Xc = batch_Xc.reshape((batch_Xc.shape[0], batch_Xc.shape[1]))
        
        batch_Ximg, batch_Xc, batch_Y = shuffle(batch_Ximg, batch_Xc, batch_Y)
        
        #Cut again to the batch size
        batch_Ximg = batch_Ximg[0:self.batch_size]
        batch_Xc = batch_Xc[0:self.batch_size]
        batch_Y = batch_Y[0:self.batch_size]
        

        return [np.array(batch_Ximg),np.array(batch_Xc)], np.array(batch_Y)