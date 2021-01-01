# Fashion Class Classification

## Problem Statement
#### The purpose of this project was to design a classification model for the MNIST fashion dataset. The dataset conations clothes of the following 10 classes:
### Classes
#### 0 = T-shirt/top
#### 1 = Trouser
#### 2 = Pullover
#### 3 = Dress
#### 4 = Coat
#### 5 = Sandal
#### 6 = Shirt
#### 7 = Sneaker
#### 8 = Bag
#### 9 = Ankle boot

## Dataset
#### Because of the file size, the dataset cannot be uploaded on Github. However, you can download it from the link below.
#### Link: https://drive.google.com/file/d/1D9_M_3mSDQkLGvWa8tfsLi8egJHyuJnO/view?usp=sharing

## Model
#### Deep Learning was used to solve this classification problem. A Convolutional Neural Network (CNN) was designed for this purpose. Two variations of the CNN were tried.

### Version 1: 32 kernels/filters in input layer
#### This version contained 32 kernels in the input layer.
#### It achieved an accuracy of 87.2% on the test set.

### Version 2:  64 kernels/filters in input layer
#### This version contained 64 kernels in the input layer.
#### It achieved an accuracy of 88.2% on the test set.

## Conclusion
#### The 64 kernel model was chosen as it gave higher validation accuracy. 
