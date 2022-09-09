#%% import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from sklearn import svm
from sklearn import metrics
import pickle
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix
#%% Read the data
df=pd.read_csv('parkinsons.data')
df.head()
#DataFlair - Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
#DataFlair - Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#DataFlair - Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
# = features
y=labels
#%% split dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
print("x_train shape")
print(x_train.shape)
print("--------------------")
print("y_train shape")
print(y_train.shape)

#%% preprocess input data
targets_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
targets_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
inputs_train = np.expand_dims(x_train, axis=2)
inputs_test = np.expand_dims(x_test, axis = 2)
print(inputs_train.shape)
print(inputs_test.shape)
#%% train SVM model 
clf = svm.SVC(kernel='linear') # Linear Kernel
y_score = clf.fit(x_train, y_train)
#%% calculate SVM accuracy 
y_pred_test = clf.predict(x_test)
print(" Linear SVM Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
y_pred = clf.predict(x_train)
print(" Linear SVM Train Accuracy:",metrics.accuracy_score(y_train, y_pred))
#%% save model 
filename = 'parkinson-gui-SVM-model.sav'
pickle.dump(clf, open(filename, 'wb'))
#%% calculate confusion matrix 
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
#%% calculate accuracy metrics 
print("  0 -> No Parkinson")
print("  1 -> Parkinson")
print("------------------------------------------------------")
print(metrics.classification_report(y_test, y_pred_test))
