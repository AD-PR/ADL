# -*- coding: utf-8 -*-
"""
@author: cmoralesb
@author: csanmiguel
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras import regularizers
import tensorflow_addons as tfa

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from Plotters import scatter_clusters
from AlzheimerFeatureExtractor import AlzheimerFeatureExtractor3D
from CustomGenerators import CustomGeneratorImg, CustomGenerator

tf.keras.backend.set_floatx('float32')
my_callbacks_val = [ tf.keras.callbacks.EarlyStopping(patience=5)]

class Parameters():
    ''' Defining general parameter to build, train and test any NN:
        
        Arg**:
            w_regularizer: Keras regularizer.
            batch_size: int
            dropout: float btw 0-1
            epochs: int
            gpu: boolean. Default: false
            image_shape: 3D shape
            clinical_shape: clinical features dimension
    '''
    def __init__ (self, param_dict):
        self.w_regularizer = param_dict['w_regularizer']
        self.batch_size = param_dict['batch_size']
        self.drop_rate = param_dict['drop_rate']
        self.epochs = param_dict['epochs']
        #self.gpu = param_dict['gpu']
        self.image_shape = param_dict['image_shape']
        self.clinical_shape = param_dict['clinical_shape']
        self.validation = param_dict['validation']
        self.val_size = param_dict['val_size']

class AlzheimerModel():
    
    
    def __init__(self, params, embedding_size, with_clinical = True):
        
        self.params = params
        self.mri = Input (shape = (self.params.image_shape))
        self.embeddingsize = embedding_size #Size of output embedding
        self.with_clinical = with_clinical
        if (self.with_clinical == True):
            self.clinical = Input (shape = (self.params.clinical_shape))
        else:
            self.clinical = None
        
        feature_extractor = AlzheimerFeatureExtractor3D(self.mri, self.clinical,
                                                        embeddingsize = self.embeddingsize,
                                                        w_regularizer = self.params.w_regularizer, 
                                                        drop_rate = self.params.drop_rate,
                                                        with_clinical = self.with_clinical)
        
        
        if (self.with_clinical == True):
            self.model_input = feature_extractor
            self.model_output = Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))(self.model_input)
            self.model = Model(inputs=[self.mri, self.clinical], outputs=self.model_output)
        else:
            self.model_input = feature_extractor
            self.model_output = Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))(self.model_input)
            self.model = Model(inputs=self.mri, outputs=self.model_output)
    
    def train_tiplet_loss(self, X_img, Y_img, X_clinical = None, X_img_val=None ,Y_img_val=None,Xc_val=None,plot = True):
        
        
        #Split in train and validation set, taking into account if clinical features exists
        
        if (self.with_clinical == True):
            
            if self.params.validation:
                #X_img, X_img_val, X_clinical, Xc_val, Y_img, Y_img_val = train_test_split(X_img, X_clinical, Y_img,
                #                                                                          test_size=self.params.val_size,
                #                                                                         random_state=42, shuffle=True)
                
                generator_val = CustomGenerator(X_img_val, Xc_val, Y_img_val, self.params.batch_size)
        

            generator_train = CustomGenerator(X_img, X_clinical, Y_img, self.params.batch_size)
        else:
            
            if self.params.validation:
                X_img, X_img_val, Y_img, Y_img_val = train_test_split(X_img, Y_img,
                                                                      test_size=self.params.val_size,
                                                                      random_state=42, shuffle=True)
                generator_val = CustomGeneratorImg(X_img_val, Y_img_val, self.params.batch_size)
            
            generator_train = CustomGeneratorImg(X_img, Y_img, self.params.batch_size)
            
        #Compile model
        lr=1e-2
        decay_rate = lr / self.params.epochs
        #optimizer = SGD(learning_rate=lr, momentum=0.9, nesterov=False)

        optimizer = SGD(learning_rate=lr, momentum=0.9, nesterov=False,decay=decay_rate)
        
        self.model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=optimizer)
        
        if self.params.validation:
            history = self.model.fit_generator(generator_train,
                                               steps_per_epoch = np.ceil(X_img.shape[0]/self.params.batch_size),
                                               epochs = self.params.epochs,
                                               verbose = 1,
                                               shuffle = True,
                                               validation_data = generator_val,
                                               validation_steps =  np.ceil(X_img_val.shape[0]/self.params.batch_size),
                                               callbacks = my_callbacks_val)
        else:
            history = self.model.fit_generator(generator_train,
                                               steps_per_epoch = np.ceil(X_img.shape[0]/self.params.batch_size),
                                               epochs = self.params.epochs,
                                               verbose = 1,
                                               shuffle = True)
            
            
        if plot == True:
            
            if (self.params.validation == True):
                
                if (self.with_clinical == True):
                    embeddings_train = self.model.predict([X_img, X_clinical])
                    embeddings_val = self.model.predict([X_img_val, Xc_val])
                else:
                    embeddings_train = self.model.predict(X_img)
                    embeddings_val = self.model.predict(X_img_val)
                
                projection_train =  TSNE(n_components=2).fit_transform(embeddings_train)
                projection_val =  TSNE(n_components=2).fit_transform(embeddings_val)
                scatter_clusters(projection_train, Y_img, "Training Data Clusters")
                scatter_clusters(projection_val, Y_img_val, "Validation Data Clusters")
            
            else:
                
                if (self.with_clinical == True):
                    embeddings_train = self.model.predict([X_img, X_clinical])
                else:
                    embeddings_train = self.model.predict(X_img)
                    
                projection_train =  TSNE(n_components=2).fit_transform(embeddings_train)
                scatter_clusters(projection_train, Y_img, "Training Data Clusters")
        
        return history.history

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
    validation = True
    val_size = 0.2
    clinical_dim = 20
    
    
    params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
                   'drop_rate': drop_rate, 'epochs': epochs, 
                   'image_shape': (width, high, depth, channels),
                   'clinical_shape': (clinical_dim, ),
                   'validation': validation, 'val_size': val_size}
    
    params = Parameters(params_dict)
    
    #Try with_clinical = True
    ADP1 = AlzheimerModel(params, embedding_size, with_clinical = True)
    ADP1.model.summary()
    
    #Try with_clinical = False
    ADP2 = AlzheimerModel(params, embedding_size, with_clinical = False)
    ADP2.model.summary()
    
        
        
        
        
        
        
        
        
        
        
        
        
