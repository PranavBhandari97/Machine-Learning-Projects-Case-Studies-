# -*- coding: utf-8 -*-

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

#Importing the dataset
dataset = pd.read_csv('Financial-Data.csv')

#Exploring the dataset
dataset.head()
dataset.tail()
dataset.columns
len(dataset.columns)
len(dataset)
dataset.info()
dataset.describe()
dataset.isna().sum()

#Feature Engineering
#The months_employed column has a large number of 0 values with any reason.
#This can have negative effects on our model. 
#Thus, dropping that column.
dataset = dataset.drop(columns = ['months_employed'])
#Check
dataset.columns

#The number of years a user is using an account is split into two columns, years and months. 
#Thus, combining those columns to make a single column containing the total number of months a user has been with the bank.
dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
#Check
dataset[['personal_account_m', 'personal_account_y', 
         'personal_account_months']].head()

#Dropping columns
dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])
#Check
dataset.columns

#Encoding the categorical data
dataset = pd.get_dummies(dataset)
#Check
dataset.columns

#Removing a column avoid dummy variable trap
dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])

#Creating separate datasets for Response and Users as they wont be a part of the training data
response = dataset["e_signed"]
users = dataset['entry_id']
dataset = dataset.drop(columns = ["e_signed", "entry_id"])

#Building the training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, 
                                                    response, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

#Feature Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

#Model Building
#Building and comparing various models to see the best fitting one
#Model 1: L1 Regularization (Logistic Regression)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)

#Predicting the results obtained using Logistic Regression
y_pred = classifier.predict(X_test)

#Computing the results(Logistic Regression)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:',acc)
print('Precision:',prec)
print('Recall:',rec)
print('f1_score:',f1)

#Creating a dataframe to store all the results and compare at end
results = pd.DataFrame([['Logistic Regression(L1)', acc, prec, rec, f1]],
                       columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
#Check
results

#Model 2: Support Vector Machine(SVM)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'linear')
classifier.fit(X_train, y_train)

#Predicting the results obtained using SVM
y_pred = classifier.predict(X_test)

#Computing the results(SVM)
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:',acc)
print('Precision:',prec)
print('Recall:',rec)
print('f1_score:',f1)

#Appending the values to the results dataframe
model_results = pd.DataFrame([['SVM', acc, prec, rec, f1]],
                       columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True)
#Check
results

#Model 3: Kernel SVM
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

#Predicting the results obtained using Kernel SVM
y_pred = classifier.predict(X_test)

#Computing the results(Kernel SVM)
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:',acc)
print('Precision:',prec)
print('Recall:',rec)
print('f1_score:',f1)

#Appending the values to the results dataframe
model_results = pd.DataFrame([['Kernel SVM', acc, prec, rec, f1]],
                             columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True)
#Check
results

#Model 4: Random Forest(n=100)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)

#Predicting the results obtained using Random Forest(n=100)
y_pred = classifier.predict(X_test)

#Computing the results(Random Forest)
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:',acc)
print('Precision:',prec)
print('Recall:',rec)
print('f1_score:',f1)

#Appending the values to the results dataframe
model_results = pd.DataFrame([['Random Forest(n=100)', acc, prec, rec, f1]],
                             columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True)
#Check
results

#Thus, it is clear that Random Forest is giving the best results.
#Applying kfold to validated the accuracy of Random Forest
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train,
                             y = y_train, cv = 10)
print("Random Forest Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))

#Kfold validates that Random Forest gives us the best accuracy.
#Thus, now applying Grid Search for Parameter Tuning.
#Trial 1 : Gini
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

#Predicting results using grid search and adding the results to results dataframe to see performance improvement
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest(Tuned with Gini)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
#Check
results

#Trial 2 : Entropy
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

#Predicting results using grid search and adding the results to results dataframe to see performance improvement
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest(Tuned with Entropy)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
#Check
results

#Building the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Final results with User id, Response and predicted response
final_results = pd.concat([y_test, users], axis = 1).dropna()
final_results['predictions'] = y_pred
final_results = final_results[['entry_id', 'e_signed', 'predictions']]





