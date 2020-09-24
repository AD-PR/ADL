
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from AlzheimerModel import Parameters, AlzheimerModel


xtrain=  '.../TrainingData/X_train.npy'
xtest =  '.../TrainingData/X_test.npy'
xval =  '.../TrainingData/X_val.npy'


ytrain=  '.../TrainingData/Y_train.npy'
ytest =  '.../TrainingData/Y_test.npy'
yval =  '.../TrainingData/Y_val.npy'

clinicaltrain =  '.../TrainingData/Clinical_train.npy'
clinicaltest =  '.../TrainingData/Clinical_test.npy'
clinicalval =  '.../TrainingData/Clinical_val.npy'

X_train = np.load(xtrain)
X_test = np.load(xtest)
X_val = np.load(xval)

Y_train = np.load(ytrain)
Y_test = np.load(ytest)
Y_val = np.load(yval)

Clinical_train = np.load(clinicaltrain)
Clinical_test = np.load(clinicaltest)
Clinical_val = np.load(clinicalval)



#Training Parameters
drop_rate = 0.1
w_regularizer = regularizers.l2(0.05)
batch_size = 64
epochs = 100
embedding_size = 200
width = X_train.shape[1]
high = X_train.shape[2]
depth = X_train.shape[3]
channels = 1 #Black and white
validation = True
val_size = 0.1
clinical_dim = 27   


params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
               'drop_rate': drop_rate, 'epochs': epochs, 
               'image_shape': (width, high, depth, channels),
               'clinical_shape': (clinical_dim, ),
               'validation': validation, 'val_size': val_size}
params = Parameters(params_dict)



# Create the model
ADP = AlzheimerModel(params, embedding_size, with_clinical = True)
#or 
#ADP = AlzheimerModel(params, embedding_size, with_clinical = False)
ADP.model.summary()

#Draw model
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(ADP.model, to_file=dot_img_file, show_shapes=True)

# Train with data
ADP.train_tiplet_loss(X_train, Y_train, X_clinical = Clinical_train,X_img_val=X_val,
                      Y_img_val=Y_val,Xc_val=Clinical_val, plot = True)


#Final Clasiffication with SVM and Metrics

train_x_embeddings = ADP.model.predict([X_train, Clinical_train])
test_x_embeddings = ADP.model.predict([X_test, Clinical_val])
val_x_embeddings = ADP.model.predict([X_val, Clinical_test])

svc = SVC()
svc.fit(train_x_embeddings, Y_train)
test_prediction = svc.predict(test_x_embeddings)
print("test accuracy : ", accuracy_score(Y_test, test_prediction))
train_prediction = svc.predict(train_x_embeddings)
print("train accuracy : ", accuracy_score(Y_train, train_prediction))
val_prediction = svc.predict(val_x_embeddings)
print("val accuracy : ", accuracy_score(Y_val, val_prediction))

tn, fp, fn, tp = confusion_matrix(Y_val, test_prediction).ravel()

sensitivity = tp / (tp+fn)
print("sensitivity : ", sensitivity)
specificity = tn / (tn+fp)
print("specificity : ", specificity)
False_pr  = 1-specificity
print("False positive rate : ", False_pr)
False_nr = 1 - sensitivity
print("False negative rate : ", False_nr)


#Save model
tf.keras.models.save_model(ADP.model, 'model_batch64_lr1e3_100embedding', overwrite=True,
                           include_optimizer=True, save_format=True)
