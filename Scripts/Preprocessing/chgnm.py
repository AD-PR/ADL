# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:09:55 2020

@author: cmartinez
"""

import os
import sys
import shutil as shtl
import pandas as pd
import numpy as np
import math

adni = pd.read_csv('../ADNI/ADNIMERGE.csv', delimiter=',')
path = '/home/cmartinez/Projects/ADELE/Dataset/'
subpath = 'ADNI'

values = {'CN' : 'NL', 'MCI' : 'MCI', 'EMCI' : 'MCI', 'LMCI' : 'MCI', 'Dementia' : 'AD', 'AD' : 'AD', 'SMC' : 4}

for r, d, f in os.walk(path+subpath):
    for file in f:
        if file[5:15] in pd.unique(adni['PTID']):
            if not os.path.exists(path+file[5:15]):
                os.makedirs(path+file[5:15])
            fname = file[0:64]+'.nii'
            #os.rename(r+'/'+file, r+'/'+fname)
            shtl.copy2(r+'/'+file, path+file[5:15])
            rows = adni.index[adni['PTID'] == file[5:15]].tolist()
            initstate = values[adni.loc[adni.index[adni['PTID'] == file[5:15]].tolist()]['DX_bl'].tolist()[0]]
            endstate  = adni.loc[adni.index[adni['PTID'] == file[5:15]].tolist()]['DX'].tolist()
            for ii in range(len(endstate)):
                if type(endstate[ii]) !=  str:
                    if math.isfinite(endstate[ii]) :
                        valendstate = valendstate
                else:
                    valendstate = values[endstate[ii]]
            # As my files are only MRI images, they are named as _masked_brain.nii
            # If they were JD, they must be named as JD_masked_brain.nii
            if initstate == valendstate:
                varnm = 'stable'+initstate+'_'+file[5:15]+'_MPRAGE_masked_brain.nii'
            else:
                varnm = 'stable'+initstate+'to'+valendstate+'_'+file[5:15]+'_MPRAGE_masked_brain.nii'
            if os.path.exists(path+file[5:15]+'/'):
                print('PATH EXISTS')
                os.rename(path+file[5:15]+'/'+file, path+file[5:15]+'/'+varnm)

        else:
            print(file)
