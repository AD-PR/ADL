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
output_file("trial2.html")
#data = {'full': [img_data[:,:,:,0]], 'partial': [img_data[35,:,:,0].T]}
full = img_data[:,:,:,0]
partial = img_data[35,:,:,0]
#zsrc = ColumnDataSource(data=dict(a=img_data[0,:,:,0]))
zsrc = ColumnDataSource(data=dict(partial=partial, full=full))
color_mapper = LinearColorMapper(bokeh.palettes.viridis(256))
plot2 = figure(tools='', toolbar_location=None,title="Sliders example", match_aspect=True)    
#plot2.image(image=[img_data[35,:,:,0].T], x=0, y=0, dw=nj, dh=nk, color_mapper=color_mapper)
plot2.image('partial', source=zsrc, x=0, y=0, dw=nj, dh=nk, color_mapper=color_mapper)

x_slider = Slider(start=0, end=ni, value=35, step=1, title='X')
callback = CustomJS(args=dict(source=zsrc, xsl=x_slider),  code="""
        const data = source.data;
        var full = data['full']
        var partial = data['partial']
        var xslv = x_slider.value
        data['partial'] = full[:,:,xslv].T
        source.change.emit();
""")




#callback = CustomJS(args=dict(source=zsrc, xsl=x_slider),
#                        code="""
#        var data = source.data;
#        var full = data['full']
#        var partial = data['partial']
#        const xslv = xsl.value
#        partial = full[:,:,xslv]
#        source.change.emit();
#    """)
    
x_slider.js_on_change('value', callback)
widgets = column(x_slider)
output_file("trial3.html")
layout = column(x_slider, plot2)
#l = grid([column(widgets, plot2)], sizing_mode='stretch_both')
show(layout)




#def slider(img_data):
#    #color_mapper = LinearColorMapper(bokeh.palettes.RdBu(256))
#    
#    n_i, n_j, n_k, n_l = img_data.shape
#    slice_0 = img_data[0, :, :, 0]
#    slice_1 = img_data[:, 0, :, 0]
#    slice_2 = img_data[:, :, 0, 0]
#    
#    source = ColumnDataSource(data=dict(a=img_data[:,:,:,0]))
#
#    plot2 = figure(tools='', toolbar_location=None,title="Sliders example", match_aspect=True)    
#    plot2.image(image=[slice_0.T], x=0, y=0, dw=n_j, dh=n_k)
#
#    x_slider = Slider(start=0, end=n_i, value=0, step=1, title='X')
#    #y_slider = Slider(start=0, end=n_j, value=0, step=1, title='Y')
#    #z_slider = Slider(start=0, end=n_z, value=0, step=1, title='Z')
#    
#    callback = CustomJS(args=dict(source=source, xsl = x_slider),
#                        code="""
#        const data = source.data;
#        const xslv = xsl.value
#        
#        data = data[xslv,:,:]
#        source.change.emit();
#    """)
#    
#    
#    x_slider.js_on_change('value', callback)
#    
#    widgets = column(x_slider)
#    return [widgets, plot2]
#
#    
#    #amp_slider = Slider(start=0, end=10, value=1, step=.1, title="Amplitude")
#    #freq_slider = Slider(start=0, end=10, value=1, step=.1, title="Frequency")
#    #phase_slider = Slider(start=0, end=6.4, value=0, step=.1, title="Phase")
#    #offset_slider = Slider(start=0, end=5, value=0, step=.1, title="Offset")
#
#
#
##    callback = CustomJS(args=dict(source=source, amp=amp_slider, freq=freq_slider, phase=phase_slider, offset=offset_slider),
##                        code="""
##        const data = source.data;
##        const A = amp.value;
##        const k = freq.value;
##        const phi = phase.value;
##        const B = offset.value;
##        const x = data['x']
##        const y = data['y']
##        for (var i = 0; i < x.length; i++) {
##            y[i] = B + A*Math.sin(k*x[i]+phi);
##        }
##        source.change.emit();
##    """)
#
#    #amp_slider.js_on_change('value', callback)
#    #freq_slider.js_on_change('value', callback)
#    #phase_slider.js_on_change('value', callback)
#    #offset_slider.js_on_change('value', callback)
#
#    #widgets = column(amp_slider, freq_slider, phase_slider, offset_slider)
#    #return [widgets, plot1]


