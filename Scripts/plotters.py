# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:52:09 2020

@author: carlosm
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

def scatter_clusters(x, labels, subtitle=None):
    
    # Color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # Scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    #Labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.savefig(subtitle)