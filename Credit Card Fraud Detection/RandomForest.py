# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 08:32:07 2021

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

#Building the model
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train.values.ravel())

#Predicting the results
y_pred = random_forest.predict(X_test)

#Building the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Calculating the score
random_forest.score(X_test,y_test)
