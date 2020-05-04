# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:53:44 2020

@author: cmorales
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, BatchNormalization, ELU
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from sklearn.manifold import TSNE
from plotters import scatter_clusters


class Parameters():
    ''' Defining general parameter to build, train and test any NN:
        
        Arg**:
            w_regularizer: Keras regularizer.
            batch_size: int
            dropout: float btw 0-1
            epochs: int
            gpu: boolean. Default: false
            image_shape: 3D shape
    '''
    def __init__ (self, param_dict):
        self.w_regularizer = param_dict['w_regularizer']
        self.batch_size = param_dict['batch_size']
        self.drop_rate = param_dict['drop_rate']
        self.epochs = param_dict['epochs']
        #self.gpu = param_dict['gpu']
        self.image_shape = param_dict['image_shape']  

class FeatureExtractor3D():
    
    
    def __init__(self, params, embedding_size):
        
        self.params = params
        self.mri = Input (shape = (self.params.image_shape))
        self.embeddingsize = embedding_size #Size of output embedding
        
        
        #ConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
        #        padding = 'same', w_regularizer = None,
        #        drop_rate = None, name = None, pool = False):
        convb1 = ConvBlock3D(24, 10, 12, 10, strides = 2, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (self.mri)
        convb2 = ConvBlock3D(48, 5, 6, 5, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=False) (convb1)
        convb3 = ConvBlock3D(96, 5, 6, 5, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=False) (convb2)
        convb4 = ConvBlock3D(48, 5, 6, 5, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb3)
        convb5 = ConvBlock3D(24, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=False) (convb4)
        convb6 = ConvBlock3D(8, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb5)
        flat7 = Flatten()(convb6)
        den8 = Dense(self.embeddingsize, activation=None, kernel_regularizer=self.params.w_regularizer, 
                     kernel_initializer='he_uniform')(flat7)
        norm_layer = Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))(den8)
        self.model = Model(inputs=self.mri, outputs=norm_layer)
    
    def train_tiplet_loss(self, train_X, train_Y, plot = True):
        
        optimizer = Adam(lr=0e-3) 
        self.model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=optimizer)
        
        history = self.model.fit(train_X, train_Y, batch_size=self.params.batch_size, 
                                 epochs=self.params.epochs, verbose=1)
        
        embeddings = self.model.predict(train_X)
        projections =  TSNE(n_components=2).fit_transform(embeddings)
        
        if plot == True: scatter_clusters(projections, train_Y, "Training Data Clusters")
        
        return history.history
    
    
        
    
        
def ConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
                padding = 'same', w_regularizer = None,
                drop_rate = None, name = None, pool = False):
    
    '''Defines a 3D Convolutional Block model, returning that model'''
    def f(input):
        
        ''' Arg**:
            input = 5D tensor (batch, conv_dim1, conv_dim2, conv_dim3, channels)
            '''
        conv = Conv3D(filters, (height, width, depth),
                      strides = strides, kernel_initializer="he_normal",
                      padding=padding, activation="linear",
                      kernel_regularizer = w_regularizer, name = name)(input)
        norm = BatchNormalization()(conv)
        elu = ELU()(norm)
        if pool == True:       
            elu = MaxPooling3D(pool_size=3, strides=2) (elu)
        return Dropout(drop_rate) (elu)
   
    return f

if __name__ == "__main__":
    
    drop_rate = 0.1
    w_regularizer = regularizers.l2(5e-5)
    batch_size = 20
    embedding_size = 200
    #resolution
    width = 197
    high = 233
    depth = 189
    channels = 1 #Black and white
    
    
    params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
               'drop_rate': drop_rate, 'epochs': 50, 
               'image_shape': (width, high, depth, channels)}
    
    params = Parameters(params_dict)
    extractor = FeatureExtractor3D(params, embedding_size)
    extractor.model.summary()
    #The we could call train triplet loss with a Train_X, Train_Y data
    
        
        
        
        
        
        
        
        
        
        
        
        
