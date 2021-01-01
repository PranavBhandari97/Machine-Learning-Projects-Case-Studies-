# Churn Rate Minimization

## Business Problem
#### The dataset is for the customer base of a Fintech company. The Fintech company offers a subscription to which all the customers in the database are subscribed. The goal of the project is to build a model that can determine which customer is going to unsubscribe.

## Model
#### While exploring the dataset, it was seen that the dataset contained null values. Three columns, namely age, credit_score, and rewards_earned, had na entries. As age had only limited null values, those rows were dropped. On the other hand, credit_score and rewards_earned had a lot of null values, thus, the entire column was dropped.
#### After that histograms and pie charts were plotted to see if there was any irregularity in the data. This was followed by plotting a correlation plot and then a heat map. It was seen that there was a strong correlation in some variables, thus, a column was dropped.
#### This cleaned dataset way then saved, and the model was built.
#### In the model building phase, the categorical variables were first encoded.  This was followed by creating the training and test sets. After this, normalization was performed, followed by feature scaling.
#### A Logistic Regression model was built, for classification purpose. An accuracy of 60% was achieved, which was cross-validated using kfold cross-validation, which gave an accuracy of 64.3%.
#### To achieve better accuracy, feature selection was performed. The top 10 features were selected. However, even with this, a cross-validation accuracy of 64.5% was only achieved.
#### This was followed by combining the user id, the actual churn, and the predicted churn.

## Conclusion
#### Timeframe of churn was not there in the dataset. Thus, we cannot say when a customer will churn. However, the model can be used, and the customers who are likely to churn can be given extra features, or new features can be built by considering their thoughts (this can be done through polls). After adding these features, the model can be retrained with the new features, and the churn rate can be re-analyzed.
