#%% import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
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
#%% train xgboost model 
model=XGBClassifier()
model.fit(x_train,y_train)
#%% calculate test accuracy
y_pred_test=model.predict(x_test)
print("\n XGBOOST Test Accuracy", accuracy_score(y_test, y_pred_test)*100)
#print(y_pred_test)
#%% calculate train accuracy 
y_pred = model.predict(x_train)
print("\n XGBOOST Train Accuracy" ,accuracy_score(y_train, y_pred))
#print(y_pred)
#%% calculate confision matrix 
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
#%% calculate accuracy metrics 
print("  0 -> No Parkinson")
print("  1 -> Parkinson")
print("------------------------------------------------------")
print(metrics.classification_report(y_test, y_pred_test))
