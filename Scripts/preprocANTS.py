#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:39:28 2020

@author: cmartinez
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import sys
import os
import numpy as np
import ants
import nibabel as nib
import matplotlib as mlib
import matplotlib.pyplot as plt

plt.close('all')
#from nibabel.testing import data_path
# --------------------------------------------------
#import holoviews as hv
#from holoviews import opts
#hv.extension('bokeh')
# --------------------------------------------------

class PreprocessMRI:
    
    
    def __init__(self, method = 'single', path = 'none', file = 'none', tex = True):
        self.method = method
        self.path   = path
        self.file   = file
        
        self.tex = tex
        if self.tex==True:
            mlib.rc('text', usetex = True)
            mlib.rc('font', family = 'serif')
        else:
            mlib.rc('text', usetex = False)
            
    def gettemplate(self):
        tmpldir = '/home/cmartinez/Projects/ADELE/Dataset/Template/'
        fnm = 'mni_icbm152_t1_tal_nlin_sym_09a.nii'
        self.tmplimg = ants.image_read(tmpldir+fnm)
        
        return self.tmplimg
        
    def getniiimages(self):
        try:
            if self.method == 'single':
                self.niimage = nib.load(self.path+self.file)
                self.niimagedata = self.niimage.get_fdata()
                
            elif self.method == 'full':
                # Not working yet
                self.niimagearray= []
                for r, d, f in os.walk(self.path):
                    for file in f:
                        self.niimagearray.append(r+'/'+file)
                #niimage =
        except:
            print('Method failed to extract niimagedata')             
    
        return self.niimage, self.niimagedata
    
    def getantsimg(self):
        try:
            if self.method == 'single':
                # Working
                self.antsimg = ants.image_read(self.path+self.file)
                return self.antsimg
                
            elif self.method == 'some':
                # Not working yet
                self.antsimagarray = []
                print(self.file)
                for f in self.file:
                    print(f)
                    self.antsimgarray.append(ants.image_read(self.path+f))
                    print(self.antsimgarray)
                    
            elif self.method == 'full':
                # Not working yet
                self.imgarray = []
                for r, d, f in os.walk(self.path):
                    for file in f:
                        self.niimagearray.append(r+'/'+file)
        except:
            print('Method failed to extract antsimagedata')
            
        
    
def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices), figsize=(13,6))
        for i, slice in enumerate(slices):
            #axes[i].imshow(slice.T, cmap="gray", origin="lower")
            axes[i].imshow(slice.T, origin="lower")
    
#! -----------------------------------------------------------------

path = '/home/cmartinez/Projects/ADELE/Dataset/'
file = 'stableAD/stableAD_002_S_0619_MPRAGE_masked_brain.nii'
file2 = 'stableMCItoAD/stableMCItoAD_002_S_0729_MPRAGE_masked_brain.nii'

x1 = PreprocessMRI(method='single', path=path, file=file)
x2 = PreprocessMRI(method='single', path=path, file=file2)
img1 = x1.getantsimg()
img2 = x2.getantsimg()
tmplimg = x1.gettemplate()

img2 = img2.n3_bias_field_correction( 8 ).n3_bias_field_correction( 4 )
mask2 = ants.get_mask(img2, low_thresh = img2.mean() * 1.1, high_thresh=1e9, cleanup = 5 ).iMath_fill_holes()
masktmpl = ants.get_mask(tmplimg, low_thresh=tmplimg.mean()*0.75, high_thresh=1e9, cleanup = 3).iMath_fill_holes()

t1rig = ants.registration( tmplimg * masktmpl, img2 * mask2, "BOLDRigid" )
t1reg = ants.registration( tmplimg * masktmpl, img2 * mask2, "ElasticSyN",
  initialTransform = t1rig['fwdtransforms'],
  synMetric = 'CC', synSampling = 2, regIterations = (5) )

ants.plot(tmplimg, overlay_alpha=0.5, axis=0, ncol=10, nslices=10*4)
ants.plot( tmplimg*masktmpl, t1reg['warpedfixout'] , axis=0, overlay_alpha=0.25, ncol=10, nslices=10*4 )
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











