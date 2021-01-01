# Credit Card Fraud Detection

## Business Problems
#### The dataset is for an E-commerce website. The goal of the project is to build a model that can determine whether a transaction is fraudulent or non-fraudulent.

## Dataset
#### Due to size restrictions, the dataset could not be uploaded here. However, it can be downloaded from the link given below.
#### Link: https://drive.google.com/file/d/1l2kaGm_muCHNS-23pA6knWrm_OXwc3ZT/view?usp=sharing

## Model
#### For this project, three models were tried. Random Forest (100 trees), Decision Tree, and Artificial Neural Network(ANN) were implemented to try and solve this problem.
#### While exploring the dataset, it was seen that the dataset is highly unbalanced. It contains a way higher number of non-fraudulent transactions as compared to fraudulent transactions. Thus, normalization was going to be needed to be performed. Initially, the models were built without normalization, and their accuracies were noted. After that, Undersampling and Oversampling were done for the ANN model. No normalization techniques were tried in the Random Forest, and Decision Tree classifiers as the ANN model was predicting the results quite accurately.
#### Let’s talk about the Decision Tree and Random Forest models first.  After importing the dataset and basic data exploration, feature scaling was used to scale certain columns, and the unnecessary columns were dropped. This was followed by building the models. Decision Tree gave an accuracy of 99.92%, while Random Forest gave an accuracy of 99.95%.
#### After this, the ANN model was built. It had 29 nodes in the input layers, 16 nodes in the first hidden layer, and 24 nodes in the second hidden layer with a dropout rate of 0.5. An accuracy of 99.94% was achieved. I decided to normalize the data and try to refit the model using ANN only. However, the same techniques can be used with Random Forest or Decision Trees as well.

### Undersampling
#### The dataset was heavily unbalanced. When undersampling was used, and the number of non-fraudulent transactions was made equal to the number of fraudulent transactions, the model lost a lot of information. Thus, the accuracy dropped to 94.59%. The model even classified 16 fraudulent transactions as non-fraudulent which was bad.

### Oversampling
#### When oversampling was performed, the accuracy of the model still dropped. The dropped was way lesser, and the accuracy fell to 99.6%. However, the good thing with this was that the model didn’t wrong classify any fraudulent transactions. All fraudulent transactions were rightly classified, and some non-fraudulent transactions were wrongly classified, which is ok.

## Conclusion
#### Thus, the ANN trained on the oversampled dataset was decided to be used for making future predictions. The only drawback for this was that the E-commerce website would need to set up a team who can cross verify the wrongly classified non-fraudulent transactions. However, this still saves the website a lot of money as no fraud transaction could be performed.
