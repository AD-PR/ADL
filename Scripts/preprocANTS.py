#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:39:28 2020

@author: cmartinez
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import numpy as np
import ants
import nibabel as nib
import matplotlib as mlib
import matplotlib.pyplot as plt
from deepbrain import Extractor
#np.random.seed(123)  # for reproducibility
#import tensorflow as tf

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from tensorflow.keras.datasets import mnist
#from tensorflow.compat.v1 import InteractiveSession
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session()
#session = InteractiveSession(config=config)

plt.close('all')

class PreprocessMRI:
    
    
    def __init__(self, method = 'single', fullfnm = 'none', tex = True):
        self.method = method
        self.fullfnm = fullfnm
        
        self.tex = tex
        if self.tex==True:
            mlib.rc('text', usetex = True)
            mlib.rc('font', family = 'serif')
        else:
            mlib.rc('text', usetex = False)
            
    def gettemplate(self):
        tmpldir = '/home/cmartinez/Projects/ADELE/Dataset/Template/'
        fnm = 'mni_icbm152_t1_tal_nlin_sym_09a.nii'
        fnmmask = 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii'
        self.tmplimg = ants.image_read(tmpldir+fnm)
        self.tmplimgmask = ants.image_read(tmpldir+fnmmask)
        self.im = self.tmplimg.new_image_like(np.multiply(self.tmplimg.numpy(), self.tmplimgmask.numpy()))
        
        return self.im
        
    def getniiimages(self):
        try:
            if self.method == 'single':
                self.niimage = nib.load(self.fullfnm)
                self.niimagedata = self.niimage.get_fdata()
                
        except:
            print('Method failed to extract niimagedata')             
    
        return self.niimage, self.niimagedata
    
    def getantsimg(self):
        try:
            if self.method == 'single':
                # Working
                self.antsimg = ants.image_read(self.fullfnm)
                return self.antsimg
                                    
        except:
            print('Method failed to extract antsimagedata')
            
    
    def get_brain_antsimg(self, ext='none'):
        try:
            if self.method == 'single':
                self.antsimg = ants.image_read(self.fullfnm)
                self.prob    = ext.run(self.antsimg.numpy())
                self.mask    = self.prob > 0.5
                self.tmp     = np.multiply(self.antsimg.numpy(), self.mask)
                self.im      = self.antsimg.new_image_like(self.tmp)
                
                return self.im
                
        except:
            print('Method failed to extract skull-stripped image')
        
    
def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices), figsize=(13,6))
        for i, slice in enumerate(slices):
            #axes[i].imshow(slice.T, cmap="gray", origin="lower")
            axes[i].imshow(slice.T, origin="lower")
    
#! -----------------------------------------------------------------

tmpl = PreprocessMRI()
imgtmpl = tmpl.gettemplate()
#imgtmpl = x1.gettemplate()
ext = Extractor()
path = '/home/cmartinez/Projects/ADELE/Dataset/'

nfiles = 1
for r, d, f in os.walk(path):
    for file in f:
       nfiles += 1

print('There are %d files' % nfiles)
count = 1
for r, d, f in os.walk(path):
    for file in f:
        if not os.path.exists(path+'Preprocessed'+'/'+file):
            print('File %d out of %d' % (count, nfiles))
            x1 = PreprocessMRI(method='single', fullfnm = r+'/'+file)
            print(r+'/'+file)
            img = x1.get_brain_antsimg(ext) 
            img = img.n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
            t1rig = ants.registration( imgtmpl, img, "AffineFast" )
            t1reg = ants.registration( imgtmpl, img, "ElasticSyN", initialTransform = t1rig['fwdtransforms'],
                                  synMetric = 'CC', synSampling = 2, regIterations = (5) )
            wrpimg = ants.apply_transforms( fixed=imgtmpl, moving=img, transformlist=t1reg['fwdtransforms'] )
            ants.image_write(wrpimg, path+'Preprocessed'+'/'+file)
        print('%d Files remaining' % (nfiles-count))
        count += 1

# file = 'stableAD_002_S_0619_MPRAGE_masked_brain.nii'
# x1 = PreprocessMRI(method='single', fullfnm = path+'/stableAD/'+file)
# img = x1.get_brain_antsimg(ext) 
# ants.plot(img, overlay_alpha=0.5, axis=0, ncol=10, nslices=10*4)
# img = img.n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )

# ants.plot(img, overlay_alpha=0.5, axis=0, ncol=10, nslices=10*4)
# ants.plot(imgtmpl, overlay_alpha=0.5, axis=0, ncol=10, nslices=10*4)

# t1rig = ants.registration( imgtmpl, img, "AffineFast" )
# t1reg = ants.registration( imgtmpl, img, "ElasticSyN", initialTransform = t1rig['fwdtransforms'],
#                       synMetric = 'CC', synSampling = 2, regIterations = (5) )
# wrpimg = ants.apply_transforms( fixed=imgtmpl, moving=img, transformlist=t1reg['fwdtransforms'] )

# ants.plot(wrpimg, axis=2, overlay_alpha=0.25, ncol=10, nslices=10*4 )
# ants.plot(imgtmpl, wrpimg , axis=2, overlay_alpha=0.25, ncol=10, nslices=10*4 )

# ants.image_write(wrpimg, path+'Preprocessed'+'/'+file)




#img_seg = ants.atropos(a=wrpimg, x=ants.get_mask(wrpimg, low_thresh=0.01))
#print(img_seg.keys())
#ants.plot(img_seg['segmentation'])

#mask2 = ants.get_mask(img2, low_thresh = img2.mean() * 0.75, high_thresh=1e9, cleanup = 5 ).iMath_fill_holes()
#masktmpl = ants.get_mask(imgtmpl, low_thresh=imgtmpl.mean()*0.75, high_thresh=1e9, cleanup = 3).iMath_fill_holes()
#ants.plot(imgtmpl*masktmpl, overlay_alpha=0.5, axis=0, ncol=10, nslices=10*4)
#ants.plot(img2*mask2, overlay_alpha=0.5, axis=0, ncol=10, nslices=10*4) 

    

#img_seg = ants.atropos(a=img2, m='[0.2,1x1]', c='[2,0]', 
#                        i='kmeans[3]', x=mask)
#print(img_seg.keys())
#ants.plot(img_seg['segmentation'])


# ni, nj, nk = img_data.shape
# center_i = (ni - 1) // 2  # // for integer division
# center_j = (nj - 1) // 2
# center_k = (nk - 1) // 2
       
# slice_0 = img_data[center_i,        :,        :]
# slice_1 = img_data[:,        center_j,        :]
# slice_2 = img_data[:,               :, center_k]
# show_slices([slice_0, slice_1, slice_2])
# plt.suptitle("Center slices for EPI image")  



# ants_img = ants.from_numpy(slice_0, has_components=False)
# mask = ants.get_mask(ants_img)
# img_seg = ants.atropos(a=ants_img, m='[0.2,1x1]', c='[2,0]', 
#                        i='kmeans[3]', x=mask)
# print(img_seg.keys())
# ants.plot(img_seg['segmentation'])

# mask = ants.get_mask( ants_img ).threshold_image( 1, 2 )
# segs=ants.atropos( a = ants_img, m = '[0.2,1x1]', c = '[2,0]',  i = 'kmeans[3]', x = mask )
# thickimg = ants.kelly_kapowski(s=segs['segmentation'], g=segs['probabilityimages'][1],
#                             w=segs['probabilityimages'][2], its=45, 
#                             r=0.5, m=1)
# print(thickimg)
# ants_img.plot(overlay=thickimg, overlay_cmap='jet')











