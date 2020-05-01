# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:53:44 2020

@author: cmorales
"""

import pandas as  pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Conv3D, MaxPooling3D, Dropout, BatchNormalization, ELU
from keras.layers import Reshape, Dense, Flatten, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Layer
from sklearn.model_selection import train_test_split


class Parameters():
    ''' Defining general parameter to build, train and test any NN:
        
        Arg**:
            w_regularizer: string 'L1', 'L2'.
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
        self.gpu = param_dict['gpu']
        self.image_shape = param_dict['image_shape']  

class FeatureExtractor3D():
    
    
    def __init__(self, params):
        
        self.params = params
        self.mri = Input (shape = (self.params.image_shape))
        self.embeddingsize = 100
        
        convb1 = ConvBlock3D(24, 10, 12, 10, strides = 5, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (self.mri)
        convb2 = ConvBlock3D(48, 5, 6, 5, strides = 2, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb1)
        convb3 = ConvBlock3D(96, 5, 6, 5, strides = 2, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb2)
        convb4 = ConvBlock3D(48, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb3)
        convb5 = ConvBlock3D(24, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb4)
        convb6 = ConvBlock3D(12, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb5)
        flat7 = Flatten()(convb6)
        den8 = Dense(self.embeddingsize, activation=None, kernel_regularizer=self.params.w_regularizer, 
                     kernel_initializer='he_uniform')(flat7)
        self.feature_extractor = Lambda(lambda x: K.l2_normalize(x,axis=-1))(den8)
    
    def train_triplet_loss(self, data, margin = 0.2):
        '''
        Input : 
            data: training data, images and labels
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
        '''
        
        # Define the tensors for the three input images
        anchor_input = self.mri
        positive_input = self.mri
        negative_input = self.mri
        
         # Define the encoded embbedings for the three input images
        encoded_a = self.feature_extractor
        encoded_p = self.feature_extractor
        encoded_n = self.feature_extractor
        
        #And the final triplet loss layer        
        loss_layer = TripletLossLayer(alpha=margin)([encoded_a,encoded_p,encoded_n])
        
        # Connect the inputs with the outputs
        triplet_model = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    

class TripletLossLayer(Layer):
    
    '''The triplet loss function, implemented as a custom Keras layer
       Loss = max(d(A,P)−d(A,N)+margin,0) where A, P, N are anchor, positive
       and negative examples'''
       
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
    
        
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





def get_APN_triplets(data, batch_size):
    
    """
    Create batch of APN triplets with a complete random strategy
    
    Arguments:
    batch_size -- integer 
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    
    #Data Format = [[image][label]] = [[197x233x189],[1]]
    height, width, depth
    m, w, h,c = X[0].shape
    
    
    # initialize result
    triplets=[np.zeros((batch_size,h, w,c)) for i in range(3)]
    
    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]
        
        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)
        
        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]
        
        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i,:,:,:] = X[anchor_class][idx_A,:,:,:]
        triplets[1][i,:,:,:] = X[anchor_class][idx_P,:,:,:]
        triplets[2][i,:,:,:] = X[negative_class][idx_N,:,:,:]

    return triplets

def drawTriplets(tripletbatch, nbmax=None):
    """display the three images for each triplets in the batch
    """
    labels = ["Anchor", "Positive", "Negative"]

    if (nbmax==None):
        nbrows = tripletbatch[0].shape[0]
    else:
        nbrows = min(nbmax,tripletbatch[0].shape[0])
                 
    for row in range(nbrows):
        fig=plt.figure(figsize=(16,2))
    
        for i in range(3):
            subplot = fig.add_subplot(1,3,i+1)
            axis("off")
            plt.imshow(tripletbatch[i][row,:,:,0],vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(labels[i])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
