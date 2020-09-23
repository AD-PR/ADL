# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:10:38 2020

@author: Carlos Sanmiguel Vila
"""


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


filename_1 =  'M3D2.npy' ## Numpy with MRI Images
filename_2 =  'L3D2.npy' ## Numpy with Classes

filename_3 = 'Clinical2.csv' ## Clinical Data obtained from GetTabularData

images = np.load(filename_1)

labels = np.load(filename_2)

df = pd.read_csv(filename_3)

# Preproccesing of the clinical data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

num_attribs = ['AGE','PTEDUCAT','PHS','APOE4','CDRSB','ADAS11','ADAS13','MMSE',
               'RAVLT_immediate','RAVLT_learning','RAVLT_perc_forgetting','FAQ']
cat_attribs = ['PTGENDER','PTETHCAT','PTRACCAT','PTMARRY']
cat_attribs2 = ['DX_bl']

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])


df_proc = df.drop(['RID', 'PTID','VISCODE','label'], axis=1)

df_proc = full_pipeline.fit_transform(df_proc)

#Eliminate some scarce instances for training:

indx = np.where((labels!="mni") & (labels!="stableMCItoNL") & (labels!="stableADtoMCI"))
data_X = images[:,:,:,indx[0]]
data_Y = labels[indx]

clinical = df_proc[indx[0],:]


del images, labels

# Get 2 classes

encoder = LabelEncoder()
data_Y = data_Y.reshape(len(data_Y), 1)
data_Y_encoded = encoder.fit_transform(data_Y)
label_map = dict((c, i) for i, c in enumerate(encoder.classes_))
print("Codificación One Hot: ", label_map)

data_X = np.moveaxis(data_X, 3, -4) #bring instance axis to the first index

data_X -= data_X.mean(axis=(1, 2, 3), keepdims=True)
data_X /= data_X.std(axis=(1, 2, 3), keepdims=True)

#In case only processing 2 classes:
data_Y_encoded = np.where(data_Y_encoded==2,0,data_Y_encoded)
data_Y_encoded = np.where(data_Y_encoded==3,1,data_Y_encoded)

data_Y_encoded = np.where(data_Y_encoded==4,0,data_Y_encoded)
data_Y_encoded = np.where(data_Y_encoded==5,1,data_Y_encoded)

#adding the channel last
data_X = np.array((data_X,))
data_X = np.moveaxis(data_X, 0, -1)

#Obtain Train, Test, Val

X_train, X_test, Y_train, Y_test, Clinical_train, Clinical_test = train_test_split(data_X, data_Y_encoded,clinical, test_size=0.15, 
                                                    random_state=42, shuffle=True)
del data_X, data_Y, data_Y_encoded


X_test, X_val, Y_test, Y_val, Clinical_test, Clinical_val = train_test_split(X_test, Y_test,Clinical_test, test_size=0.33, 
                                                    random_state=42, shuffle=True)


np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('X_val.npy',X_val)

np.save('Y_train.npy',Y_train)
np.save('Y_test.npy',Y_test)
np.save('Y_val.npy',Y_val)

np.save('Clinical_train.npy',Clinical_train)
np.save('Clinical_test.npy',Clinical_test)
np.save('Clinical_val.npy',Clinical_val)
