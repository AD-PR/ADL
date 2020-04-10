# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:43:32 2020

@author: cmartinez
"""

import os 
import numpy as np
import nibabel as nib
from nibabel.testing import data_path

example_filename = os.path.join(data_path, 'example4d.nii.gz')
import nibabel as nib
img = nib.load(example_filename)
img_data = img.get_fdata()


import matplotlib.pyplot as plt
def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
       
slice_0 = img_data[1, :, :,0]
slice_1 = img_data[:, 30, :,0]
slice_2 = img_data[:, :, 16,0]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  


