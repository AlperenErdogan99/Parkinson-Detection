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
import tensorflow.keras as keras
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
input_shape = (inputs_train.shape[1], inputs_train.shape[2]) # 130, 13
#%% create LSTM model 

model = keras.Sequential()

    # 2 LSTM layers
model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dropout(0.05))
    # dense layer
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(16, activation='relu'))

model.add(keras.layers.Dropout(0.05))
    # output layer
model.add(keras.layers.Dense(2, activation='softmax'))
    
    # compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()
    
#%%train model
history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), batch_size=32, epochs=10)
#%% calculate test accuracy 
test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
print('\nLSTM Test accuracy:', test_acc)
#%% train accuracy 
train_loss, train_acc = model.evaluate(inputs_train, targets_train, verbose=2)
print('\nLSTM Train accuracy:', train_acc)    
#%% calculate confision matrix
rounded_predictions  = model.predict_classes(inputs_test, batch_size=32, verbose=0)
rounded_labels=np.argmax(targets_test, axis=1)
cm = confusion_matrix(rounded_labels, rounded_predictions)
plt.figure(figsize = (7,7))
sn.set(font_scale=1)
sn.heatmap(cm, annot=True) 
#%% calculate accuracy metrics
print("  0 -> No Parkinson")
print("  1 -> Parkinson")
print("------------------------------------------------------")
y_pred = model.predict(inputs_test, batch_size=32, verbose=2)
report = classification_report(targets_test, y_pred.round())
print(report)  
#%% save model 
model.save('parkinson_LSTM_model.h5')