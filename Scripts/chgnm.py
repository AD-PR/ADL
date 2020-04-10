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
path = 'c:\\Users\cmartinez\Desktop\Machine Learning\Saturdays\Project\ADNI\\'
subpath = 'ADNI1_Screening_1.5T_1'

values = {'CN' : 'NL', 'EMCI' : 'MCI', 'LMCI' : 'MCI', 'Dementia' : 'AD', 'AD' : 'AD', 'SMC' : 4}

count1 = 1
count2 = 1

for r, d, f in os.walk(path+subpath):
    for file in f:
        count1 = count1+1
        if file[5:15] in pd.unique(adni['PTID']):
            if not os.path.exists(path+file[5:15]):
                os.makedirs(path+file[5:15])
            fname = file[0:64]+'.nii'
            os.rename(r+'\\'+file, r+'\\'+fname)
            shtl.copy2(r+'\\'+fname, path+file[5:15])
            rows = adni.index[adni['PTID'] == file[5:15]].tolist()
            initstate = values[adni.loc[adni.index[adni['PTID'] == file[5:15]].tolist()]['DX_bl'].tolist()[0]]
            endstate  = adni.loc[adni.index[adni['PTID'] == file[5:15]].tolist()]['DX'].tolist()
            for ii in range(len(endstate)):
                if type(endstate[ii]) !=  str:
                    if math.isfinite(endstate[ii]) :
                        valendstate = valendstate
                else:
                    valendstate = endstate[ii]
            varnm = 'stable'+initstate+'to'+valendstate+'_'+file[5:15]+'_MPRAGE_masked_brain.nii'
            if os.path.exists(path+file[5:15]+'\\'):
                print('PATH EXISTS')
                os.rename(path+file[5:15]+'\\'+fname, path+file[5:15]+'\\'+varnm)
            #print(path+file[5:15]+'\\'+f[0])  
            #print('to')
            #print( path+file[5:15]+'\\'+varnm)
            count2 = count2+1
        else:
            print(file)
