# -*- coding: utf-8 -*-
"""
@author: cmoralesb
@author: csanmiguel
"""

from tensorflow.keras.layers import Flatten, Lambda, Add, Concatenate
from ModelBlocks3D import ConvBlock3D, SepConvBlock3D, FCBlock

### STREAM FOR 3D IMAGES AND CLINICAL DATA ###
        
def AlzheimerFeatureExtractor3D(mri_input, clinical_input,embeddingsize, w_regularizer = None, drop_rate = 0.,
                                with_clinical = True):
        
        
    #1-ConvBlock3D Layers: goal is to reduce 3D image dimensionality but keeping the number
    # of filters low,not increasing parameters number so much. Add as many block user desires. 
    
    #ConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
    #        padding = 'same', w_regularizer = None,
    #        drop_rate = None, name = None, pool = False):
    
    convb1 = ConvBlock3D(48, 5, 6, 5, strides = 3, w_regularizer = w_regularizer,
                          drop_rate = drop_rate, pool=True)(mri_input)
    convb2 = ConvBlock3D(96, 5, 6, 5, strides = 1, w_regularizer = w_regularizer,
                          drop_rate = drop_rate, pool=True)(convb1)
    
    #2-SepConv3D + Residual conection at the end:
    
    #SepConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
    #        padding = 'same', depth_multiplier = 1, w_regularizer = None,
    #        drop_rate = None, name = None, dilation_rate = (1,1,1),pool = False)
    
    residual = convb2
    
    sepfilters = 96                      

    sepconvb1 = SepConvBlock3D(sepfilters, 3, 3, 3, padding='same', depth_multiplier = 1,
                               drop_rate=drop_rate, w_regularizer = w_regularizer)(convb2)                       

    sepconvb2 = SepConvBlock3D(sepfilters, 3, 3, 3, padding='same', depth_multiplier = 1, 
                               drop_rate=drop_rate, w_regularizer = w_regularizer)(sepconvb1)

    sepconvb3 = SepConvBlock3D(sepfilters, 3, 3, 3, padding='same', depth_multiplier = 1, 
                               drop_rate=drop_rate, w_regularizer = w_regularizer)(sepconvb2)
    
    #mid = Add()([convb5, residual])
    mid = Add()([sepconvb3, residual])
    
    #3-Split convolutions in two flows (compromise between normal conv and 
    # separable convolution). 96 channels --> 2 streams of 48 channels
    
    mid1= Lambda(lambda x:x[:,:,:,:,:int(sepfilters/2)])(mid)
    mid2= Lambda(lambda x:x[:,:,:,:,:int(sepfilters/2)])(mid)
    

    convb3_left = ConvBlock3D(24, 3, 4, 3, strides = 1, w_regularizer = w_regularizer,
                              drop_rate = drop_rate, pool=False) (mid1)
    convb3_right = ConvBlock3D(24, 3, 4, 3, strides = 1, w_regularizer = w_regularizer,
                              drop_rate = drop_rate, pool=False) (mid2)
    convb4_left = ConvBlock3D(8, 3, 4, 3, strides = 1, w_regularizer = w_regularizer,
                              drop_rate = drop_rate, pool=True) (convb3_left)
    convb4_right = ConvBlock3D(8, 3, 4, 3, strides = 1, w_regularizer = w_regularizer,
                              drop_rate = drop_rate, pool=True) (convb3_right)
    
    x = Concatenate(axis=-1)([convb4_left, convb4_right])
    
    #4-Flatten 3D Images Features and reduce them to embeddings size
    
    flat_images = Flatten()(x)
    img_features = FCBlock(embeddingsize, w_regularizer = w_regularizer,
                             drop_rate = drop_rate)(flat_images)
    
    
    if (with_clinical == True):
        
        #### STREAM FOR CLINICAL FEATURES ###
        
        #1 - FC Dense block to reduce further clinical features dimensionality
        
        clinical_features = FCBlock(15, w_regularizer = w_regularizer,
                                    drop_rate = drop_rate)(clinical_input)
    
        ### COMBINE IMAGE AND CLINICAL FEATURES ###
    
        features = Concatenate()([img_features, clinical_features])
    
        ### APPLY FINAL FC BLOCK AND BUILD THE MODEL ###
    
        #embeddings = FCBlock(int(embeddingsize/2), w_regularizer = w_regularizer,
        #                     drop_rate = drop_rate)(features)
        embeddings = FCBlock(int(embeddingsize/2), w_regularizer = w_regularizer,
                             drop_rate = drop_rate)(features)
    else:
        
        #embeddings = FCBlock(int(embeddingsize/2), w_regularizer = w_regularizer,
        #                     drop_rate = drop_rate)(img_features)
        embeddings = FCBlock(int(embeddingsize/2), w_regularizer = w_regularizer,
                             drop_rate = drop_rate)(img_features)
        
    
    return embeddings