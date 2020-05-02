# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:43:32 2020

@author: cmartinez
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------
import nibabel as nib
from nibabel.testing import data_path
# --------------------------------------------------
import bokeh
from bokeh.layouts import column, grid
from bokeh.models import ColumnDataSource, CustomJS, Slider, LinearColorMapper
from bokeh.plotting import figure, output_file, show, output_notebook

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
# --------------------------------------------------

example_filename = os.path.join(data_path, 'example4d.nii.gz')
img = nib.load(example_filename)
img_data = img.get_fdata()   # Extracts the image itself

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices),figsize=(13,13))
   for i, slice in enumerate(slices):
       #axes[i].imshow(slice.T, cmap="gray", origin="lower")
       axes[i].imshow(slice.T, origin="lower")

ni, nj, nk, nl = img_data.shape
center_i = (ni - 1) // 2  # // for integer division
center_j = (nj - 1) // 2
center_k = (nk - 1) // 2
       
slice_0 = img_data[center_i,        :,        :, 0]
slice_1 = img_data[:,        center_j,        :, 0]
slice_2 = img_data[:,               :, center_k, 0]
#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for EPI image")  

#%%
#output_file("trial2.html")

full = np.transpose(img_data[:,:,:,0], (2,1,0))
partial = np.transpose(img_data[35,:,:,0],(1,0))[()]
data = {'full': [full], 'partial': [partial]}
zsrc = ColumnDataSource(data)
color_mapper = LinearColorMapper(bokeh.palettes.viridis(256))
plot2 = figure(tools='', toolbar_location=None,title="Sliders example", match_aspect=True)    
plot2.image(image='partial', source=zsrc, x=0, y=0, dw=nj, dh=nk, color_mapper=color_mapper)

x_slider = Slider(start=0, end=ni, value=35, step=1, title='X')
callback = CustomJS(args=dict(source=zsrc, xsl=x_slider),  code="""
        var data = source.data;
        var full = data['full'][0];
        var partial = data['partial'][0];
        console.log('Hola')
        var xslv = x_slider.value;
        for (i=0; i<partial.length; i++){
                for (j=0; j<partial[0].length; j++){
                        partial[i][j] = full[i][j][xslv];
                }
        }
        source.change.emit();
""")

    
x_slider.js_on_change('value', callback)
widgets = column(x_slider)
output_file("trial3.html")
layout = column(x_slider, plot2)
#l = grid([column(widgets, plot2)], sizing_mode='stretch_both')
show(layout)



