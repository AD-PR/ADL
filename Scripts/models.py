# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:53:44 2020

@author: cmorales
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa
from sklearn.manifold import TSNE
from plotters import scatter_clusters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
        self.validation = param_dict['validation']
        self.val_size = param_dict['val_size']

class FeatureExtractor3D():
    
    
    def __init__(self, params, embedding_size):
        
        self.params = params
        self.mri = Input (shape = (self.params.image_shape))
        self.embeddingsize = embedding_size #Size of output embedding
        
        
        #ConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
        #        padding = 'same', w_regularizer = None,
        #        drop_rate = None, name = None, pool = False):
        convb1 = ConvBlock3D(16, 3, 4, 3, strides = 2, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=False) (self.mri)
        convb2 = ConvBlock3D(16, 3, 4, 3, strides = 2, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, pool=True) (convb1)
        convb3 = ConvBlock3D(24, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, dilation_rate=2, pool=False) (convb2)
        convb4 = ConvBlock3D(24, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, dilation_rate=2, pool=True) (convb3)
        convb5 = ConvBlock3D(32, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, dilation_rate=4, pool=False) (convb4)
        convb6 = ConvBlock3D(32, 3, 4, 3, strides = 1, w_regularizer = self.params.w_regularizer,
                             drop_rate = self.params.drop_rate, dilation_rate=4, pool=True) (convb5)
        flat7 = Flatten()(convb6)
        den8 = Dense(self.embeddingsize, activation=None, kernel_regularizer=self.params.w_regularizer, 
                     kernel_initializer='glorot_uniform')(flat7)
        norm_layer = Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))(den8)
        self.model = Model(inputs=self.mri, outputs=norm_layer)
    
    def train_tiplet_loss(self, train_X, train_Y, plot = True):
        
        #Separates into training and validation set
        if self.params.validation:
            train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, 
                                                              test_size=self.params.val_size,
                                                              random_state=42, shuffle=True)
            generator_val = CustomGenerator(val_X, val_Y, self.params.batch_size)
        
        generator_train = CustomGenerator(train_X, train_Y, self.params.batch_size)
        
        optimizer = SGD(learning_rate=1e-3, momentum=0.9, nesterov=False)
        self.model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=optimizer)
        
        #history = self.model.fit(train_X, train_Y, batch_size=self.params.batch_size, 
        #                         epochs=self.params.epochs, verbose=1)
        
        if self.params.validation:
            history = self.model.fit_generator(generator_train,
                                               steps_per_epoch = np.ceil(train_X.shape[0]/self.params.batch_size),
                                               epochs = self.params.epochs,
                                               verbose = 1,
                                               shuffle = True,
                                               validation_data = generator_val,
                                               validation_steps =  np.ceil(val_X.shape[0]/self.params.batch_size))
        else:
            history = self.model.fit_generator(generator_train,
                                               steps_per_epoch = np.ceil(train_X.shape[0]/self.params.batch_size),
                                               epochs = self.params.epochs,
                                               verbose = 1,
                                               shuffle = True)
            
        
        embeddings = self.model.predict(train_X)
        projections =  TSNE(n_components=2).fit_transform(embeddings)
        
        if plot == True: scatter_clusters(projections, train_Y, "Training Data Clusters")
        
        return history.history
    
def ConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
                padding = 'same', w_regularizer = None,
                drop_rate = None, name = None, dilation_rate = (1,1,1), 
                pool = False):
    
    '''Defines a 3D Convolutional Block model, returning that model'''
    def f(input):
        
        ''' Arg**:
            input = 5D tensor (batch, conv_dim1, conv_dim2, conv_dim3, channels)
            '''
        conv = Conv3D(filters, (height, width, depth),
                      strides = strides, kernel_initializer="glorot_uniform",
                      padding=padding, activation="linear", dilation_rate=dilation_rate,
                      kernel_regularizer = w_regularizer, name = name)(input)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        if pool == True:       
            relu = MaxPooling3D(pool_size=3, strides=2) (relu)
        return Dropout(drop_rate) (relu)
   
    return f


class CustomGenerator(Sequence):
    
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

if __name__ == "__main__":
    
    drop_rate = 0.1
    w_regularizer = regularizers.l2(0.01)
    batch_size = 12
    embedding_size = 100
    #resolution
    width = 197
    high = 233
    depth = 189
    channels = 1 #Black and white
    epochs = 50
    validation = False
    val_size = 0.2
    
    
    params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
                   'drop_rate': drop_rate, 'epochs': epochs, 
                   'image_shape': (width, high, depth, channels),
                   'validation': validation, 'val_size': val_size}
    
    params = Parameters(params_dict)
    extractor = FeatureExtractor3D(params, embedding_size)
    extractor.model.summary()
    #The we could call train triplet loss with a Train_X, Train_Y data
    
        
        
        
        
        
        
        
        
        
        
        
        
