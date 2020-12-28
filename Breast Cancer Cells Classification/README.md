# Breast Cancer Cells Classification

## Problem Statement
#### Predicting if the cancer diagnosis is benign or malignant based on several observations/features

## Dataset
#### Name: Breast Cancer Wisconsin (Diagnostic) Data Set
#### Link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## Dataset Features
#### 30 features are used, examples:
####    - radius (mean of distances from center to points on the perimeter)
####    - texture (standard deviation of gray-scale values)
####    - perimeter
####    - area
####    - smoothness (local variation in radius lengths)
####    - compactness (perimeter^2 / area - 1.0)
####    - concavity (severity of concave portions of the contour)
####    - concave points (number of concave portions of the contour)
####    - symmetry
####    - fractal dimension ("coastline approximation" - 1)
#### Datasets are linearly separable using all 30 input features
#### Number of Instances: 569
#### Class Distribution: 212 Malignant, 357 Benign
#### Target class:
####    - Malignant
####    - Benign

## Model
#### Support Vector Machine(SVM) Classifier was used to solve this problem.
#### Initially, the data was visualized using various plots like Heat maps, Bar plots, Scatter Plot, and Pairplot for exploration purposes.
#### After this training and test sets were created, and SVM was applied.
#### However, an initial accuracy of only 64% was achieved.

## Model Improvement – Part 1
#### To increase the accuracy further, the dataset was normalized.
#### Normalization worked, and an accuracy of 98% was achieved.
#### To verify, k-fold cross-validation was used. K-fold gave an average accuracy of 94.7%, with 98% being the highest.

## Model Improvement – Part 2
#### To improve the model further, Grid Search was applied for parameter tuning purposes.
#### Using the tuned parameters, a final accuracy of 97% was achieved.
