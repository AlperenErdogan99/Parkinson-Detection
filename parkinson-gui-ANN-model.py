#%% import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import  Dropout, Flatten, Dense
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
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
#%% create ANN model
model = Sequential()
model.add(Flatten(input_shape = (inputs_train.shape[1],inputs_train.shape[2])))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(16, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(8, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(2, activation = "softmax"))
model.compile(loss= "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])
#%% train model
hist = model.fit(inputs_train, targets_train, epochs = 30, batch_size = 32) 
_, accuracy = model.evaluate(inputs_test, targets_test, batch_size = 32)
print(accuracy)


#%% save model
model.save("parkinsons_ann_model.h5")
#%% predict model 
predicted = model.predict_classes(inputs_test, batch_size=32, verbose=0)

#%% calculate test accuracy 
    
test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
print('\nTest accuracy:', test_acc)
#%% calculate train accuracy 
train_loss, train_acc = model.evaluate(inputs_train, targets_train, verbose=2)
print('\nTrain accuracy:', train_acc)    
#%% Calculate Confusion Matrix     
rounded_predictions  = model.predict_classes(inputs_test, batch_size=32, verbose=0)
rounded_labels=np.argmax(targets_test, axis=1)
cm = confusion_matrix(rounded_labels, rounded_predictions)
plt.figure(figsize = (7,7))
sn.set(font_scale=1)
sn.heatmap(cm, annot=True)
#%%Calculate Classification Report 
print("  0 -> No Parkinson")
print("  1 -> Parkinson")
print("------------------------------------------------------")
y_pred = model.predict(inputs_test, batch_size=32, verbose=2)
report = classification_report(targets_test, y_pred.round())
print(report)