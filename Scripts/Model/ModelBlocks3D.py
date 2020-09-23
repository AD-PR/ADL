# -*- coding: utf-8 -*-
"""
@author: cmoralesb
@author: csanmiguel
"""

from tensorflow.keras.layers import Conv3D, MaxPooling3D 
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU
from tensorflow.keras.layers import MaxPooling2D, Dense

from SepConv3D import SeparableConv3D


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

def SepConvBlock3D(filters, height, width, depth, strides=(1, 1, 1), 
                padding = 'same', depth_multiplier = 1, w_regularizer = None,
                drop_rate = None, name = None, pool = False):
    
    '''Defines a 3D Separable Convolutional Block model, returning that model'''
    def f(input):
        
        ''' Arg**:
            input = 5D tensor (batch, conv_dim1, conv_dim2, conv_dim3, channels)
            '''
        sepconv = SeparableConv3D(filters, (height, width, depth),
                               strides = strides, depth_multiplier= depth_multiplier,
                               kernel_initializer="glorot_uniform", padding=padding,
                               kernel_regularizer = w_regularizer, name = name)(input)
        norm = BatchNormalization()(sepconv)
        relu = ReLU()(norm)
        if pool == True:       
            relu = MaxPooling2D(pool_size=3, strides=2, padding='same') (relu)
        return Dropout(drop_rate) (relu)
   
    return f

def FCBlock(units, w_regularizer = None, drop_rate = 0., name = None):
    
    '''Defines a Fully Connected Block model, returning that model'''
    def f(input):
        fc = Dense(units = units, activation = 'linear', 
                   kernel_regularizer=w_regularizer, name = name) (input)
        fc = BatchNormalization()(fc)
        fc = ReLU()(fc)
        fc = Dropout (drop_rate) (fc)
        return fc
    
    return f
