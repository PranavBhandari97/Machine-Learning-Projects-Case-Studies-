# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:43:19 2020

@author: admin
"""

#Importing the libraries
import pandas as pd
import numpy as np
import keras
np.random.seed(2)

#Importing the dataset
data = pd.read_csv('creditcard.csv')

#Exploring the dataset
data.head()
data.tail()
data.columns
len(data.columns)
len(data)
data.describe()
data.info()
data.isna().sum()

#Feature Scaling the Amount Column
from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)
#Check
data.head()

#Droppping the time column as it is not needed
data = data.drop(['Time'],axis=1)
#Check
data.head()

#Creating the Dependent and Independent Variable
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
#Check
X.head()
y.head()

#Building the training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, 
                                                    random_state=0)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#Building an ANN
#Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',
              metrics=['accuracy'])

#Training the model
model.fit(X_train,y_train,batch_size=15,epochs=5)

#Predicting the results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), 
                      y_test.reshape(len(y_test),1)),1))

#Building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Calculating the score
score = model.evaluate(X_test, y_test)
print(score)

'''The dataset has a lot more actual transactions(0) than fraud 
transcations(1). Thus, we need to normalize our dataset. 
Using oversampling and undersampling to see how both work in our case.'''

#Undersampling
#Finding the fraud transactions
fraud_indices = np.array(data[data.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)

#Taking the non-fraudulent transcations
normal_indices = data[data.Class == 0].index

#Selecting 492 random no fraudulent transactions.
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
print(len(random_normal_indices))

#Building the new dataframe with equal fraudulent and non-fraudulent transactions
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))
under_sample_data = data.iloc[under_sample_indices,:]

#Creating the Dependent and Independent Variables
X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

#Building the training and test sets and converting them to Numpy array
X_train, X_test, y_train, y_test = train_test_split(X_undersample,
                                                    y_undersample, 
                                                    test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#Checking model summary and refitting the model
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)

#Predicting the results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), 
                      y_test.reshape(len(y_test),1)),1))

#Building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Oversampling
#Importing the library
from imblearn.over_sampling import SMOTE

#Fitting and creating the variables
X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)

#Building the training and testing dataset and converting to Numpy arrays
X_train, X_test, y_train, y_test = train_test_split(X_resample,
                                                    y_resample,
                                                    test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#Checking model summary and refitting the model
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)

#Predicting the results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), 
                      y_test.reshape(len(y_test),1)),1))

#Building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

