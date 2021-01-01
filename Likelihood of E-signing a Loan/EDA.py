# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 08:23:44 2021

@author: admin
"""

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

#Plotting histograms to visualize numerical data
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#Building a plot to visualize the correlations
dataset2.corrwith(dataset.e_signed).plot.bar(figsize = (20, 10), 
                  title = "Correlation with E Signed", 
                  fontsize = 15,rot = 45, grid = True)

#Plotting heatmap to visualize correlations.
sns.set(style="white")
corr = dataset2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, 
            center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})