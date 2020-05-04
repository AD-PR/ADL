

import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models import Parameters, FeatureExtractor3D

#Import: 3D matrix of images pixels and instance's labels

images = np.load("./data/M3D.npy")
labels = np.load("./data/L3D.npy")

dict_labels_count = Counter(labels)
df_labels_dist = pd.DataFrame.from_dict(dict_labels_count, orient='index')
df_labels_dist.plot(kind='bar')

#Eliminate some scarce instances for training:

indx = np.where((labels!="mni") & (labels!="stableMCItoNL") & (labels!="stableADtoMCI"))
data_X = images[:,:,:,indx[0]]
data_Y = labels[indx]

#Enconding String Labels into One Hot Encoder

encoder = LabelEncoder()
data_Y = data_Y.reshape(len(data_Y), 1)
data_Y_encoded = encoder.fit_transform(data_Y)
label_map = dict((c, i) for i, c in enumerate(encoder.classes_))
print("Codificación One Hot: ", label_map)

# Normalize, Shuffle, and Train-Test Splitting

data_X = np.moveaxis(data_X, 3, -4) #bring instance axis to the first index
data_X = data_X.astype('float32')
#maxmin normalization
x_min = data_X.min(axis=(1, 2, 3), keepdims=True)
x_max = data_X.max(axis=(1, 2, 3), keepdims=True)
data_X = (data_X - x_min)/(x_max-x_min)
#standard normalization
#data_X -= data_X.mean(axis=(1, 2, 3), keepdims=True)
#data_X /= data_X.std(axis=(1, 2, 3), keepdims=True)

#adding the channel last
data_X = np.array((data_X,))
data_X = np.moveaxis(data_X, 0, -1)
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y_encoded, test_size=0.3, 
                                                    random_state=42, shuffle=True)


#Training Parameters
drop_rate = 0.1
w_regularizer = regularizers.l2(5e-5)
batch_size = 21
epochs = 50
embedding_size = 200
width = data_X.shape[1]
high = data_X.shape[2]
depth = data_X.shape[3]
channels = 1 #Black and white
params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
               'drop_rate': drop_rate, 'epochs': epochs, 
               'image_shape': (width, high, depth, channels)}
params = Parameters(params_dict)



# Create the model
extractor = FeatureExtractor3D(params, embedding_size)
extractor.model.summary()

# Train with data
extractor.train_tiplet_loss(X_train, Y_train, plot = True)
