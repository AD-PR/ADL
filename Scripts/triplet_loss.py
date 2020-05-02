# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:15:49 2020

@author: carlosm
"""

import numpy as np
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Layer
import random
import matplotlib.pyplot as plt


class TripletLossLayer(Layer):
    
    '''The triplet loss function, implemented as a custom Keras layer
       Loss = max(d(A,P)−d(A,N)+margin,0) where A, P, N are anchor, positive
       and negative examples'''
       
    def __init__(self, alpha=0.2, **kwargs):
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

class LossLessTripletLossLayer(Layer):
    
    '''Variant of Triplet Loss where no margin (alpha) is needed and the'''
       
    def __init__(self, N, beta, epsilon=1e-8, **kwargs):
        
        ''' Arg:
            N  --  The number of dimension (embeddings_size)
            beta -- The scaling factor, N is recommended
            epsilon -- The Epsilon value to prevent ln(0)'''
            
        self.epsilon = epsilon
        self.N = N
        self.beta = N
        super(LossLessTripletLossLayer, self).__init__(**kwargs)
    
    def lossless_triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        p_dist = -K.log(-((p_dist) / self.beta)+1+self.epsilon)
        n_dist = -K.log(-((self.N-n_dist) / self.beta)+1+self.epsilon)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.lossless_triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class SampleTripletBatch:
    
    def __init__(self, batch_size, sample_size):
        self.batch_size = batch_size
        self.P = sample_size
        
    def get_hard_batch(self, model, train_X, train_Y):
        
        #The selected triplets can be considered moderate triplets, 
        #since they are the hardest within a small subset of the data, 
        #which is exactly what is best for learning with the triplet loss
        
        ### 1-Counting number of images per class (distribution), based on labels
        num_images_by_class = []
        for i in range(len(np.unique(train_Y))):
            num_images_by_class.append(len(train_Y[train_Y == i]))
        num_images_by_class = np.array(num_images_by_class)
        
        train_y = to_categorical(train_Y, num_classes=len(np.unique(train_Y)))
        
        ### 2-Randomly sample P images based on probability distribution of classes
        P_samples = np.random.choice(np.arange(train_y.shape[1]),
                                     p=num_images_by_class/num_images_by_class.sum(), size=self.P)
        batch_samples = []
        anchors = []
        positives = []
        negatives = []
        j = 0        

        #for each class in P_samples
        for i in P_samples:
            #find the instances of the same class
            lbl_index = np.where(train_Y == i)
            #sample 1 images of each classes
            random_index = np.random.choice(lbl_index[0], size=1)
            #batch of P images
            batch_samples.append(train_X[random_index])
            j += 1

        batch_samples = np.squeeze(batch_samples)
        batch = batch_samples.reshape((-1, batch_samples.shape[1], 
                                       batch_samples.shape[2],
                                       batch_samples.shape[3], 1))

        ### 3-Predict the embedding for each image in the sample
        pred_batch_samples = model.predict(batch)

        ## 4-Find the hardest positive and negative for each "anchor" (a) within the sample
        for i in range(len(batch_samples)):
            
            similar_class = np.where(P_samples == P_samples[i])[0]
            diff_class = np.where(P_samples != P_samples[i])[0]
            #print ('Current:', P_classes[i])
            #print ('Similar Class:', similar_class)
            #print ('Different Class:', diff_class)

            a = pred_batch_samples[i]
            anchors.append(batch_samples[i])
            #print ('Anchor predicted embedding:', a, a.shape)

            # find hardest positive
            positive_dist = []
            #print ('Positive predicted embedding:')
            for sim_index in similar_class:
                #print (pred_batch_samples[sim_index], pred_batch_samples[sim_index].shape)
                positive_dist.append(np.linalg.norm(a-pred_batch_samples[sim_index]))
            #print ('Positive:', positive_dist, similar_class[np.argmax(np.array(positive_dist))])
            positives.append(batch_samples[similar_class[np.argmax(np.array(positive_dist))]])

            # find hardest negative
            negative_dist = []
            for diff_index in diff_class:
                negative_dist.append(np.linalg.norm(a-pred_batch_samples[diff_index]))
            #print ('Negative:', negative_dist, diff_class[np.argmin(np.array(negative_dist))])
            negatives.append(batch_samples[diff_class[np.argmin(np.array(negative_dist))]])
            
       ## 5- Compose the triplets, where we would have for every anchor image of the P samples,
       # its hardest positive and its hardest negative
        anchors = np.array(anchors)
        anchors = anchors.reshape(-1, anchors.shape[-3], anchors.shape[-2], anchors.shape[-1])
        positives = np.array(positives)
        positives = positives.reshape(-1, positives.shape[-3], positives.shape[-2], positives.shape[-1])
        negatives = np.array(negatives)
        negatives = negatives.reshape(-1, negatives.shape[-3], negatives.shape[-2], negatives.shape[-1])
        self.triplets = [np.expand_dims(anchors, axis=-1), np.expand_dims(positives, axis=-1), np.expand_dims(negatives, axis=-1)]

        return np.array(self.triplets)

    def show_samples(self, anchors, positives, negatives):
        rand_index = random.sample(list(np.arange(anchors.shape[0])), 4) 
        fig, ax = plt.subplots(4, 3)
        ax[0][0].imshow(np.squeeze(anchors[rand_index[0]]))
        ax[0][1].imshow(np.squeeze(positives[rand_index[0]]))
        ax[0][2].imshow(np.squeeze(negatives[rand_index[0]]))
        ax[1][0].imshow(np.squeeze(anchors[rand_index[1]]))
        ax[1][1].imshow(np.squeeze(positives[rand_index[1]]))
        ax[1][2].imshow(np.squeeze(negatives[rand_index[1]]))
        ax[2][0].imshow(np.squeeze(anchors[rand_index[2]]))
        ax[2][1].imshow(np.squeeze(positives[rand_index[2]]))
        ax[2][2].imshow(np.squeeze(negatives[rand_index[2]]))
        ax[3][0].imshow(np.squeeze(anchors[rand_index[3]]))
        ax[3][1].imshow(np.squeeze(positives[rand_index[3]]))
        ax[3][2].imshow(np.squeeze(negatives[rand_index[3]]))
        plt.show()