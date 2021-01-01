# App Behaviour Analysis

## Business Problem
#### The provided dataset is from a Fintech company. The Fintech company has a mobile app, that has both a free version as well as a subscription or a paid version. The paid version lets the users access all their financial information in one place. When the user first joins the app, he/she gets the paid version free for the first day. The goal of this project is to build a classification model to see which users enroll and which don’t for the paid subscription.
#### Two datasets are provided by the company. The first dataset(appdata10) contains information about all the features of the app the user used. The second dataset(top_screens) gives information about the top screens that are visited by the users.

## Model
#### The EDA involved building various plots to see the trends for various features. This was followed by plotting a correlation plot and a heat map to see the correlations better.
#### This was followed by feature engineering to decide the cutoff time and to decide the top screens by merging the top_screens dataset with the appdata10.
#### After merging, it was seen that some screens (Savings1, Savings2, Saving3, etc) were there, which indicated that a user had gone till the last screen. However, this was not going to be the case for a lot of the audience. Thus, funnels were created to remove this.
#### This cleaned dataset was then saved as ‘new_appdata10’
#### The model was then built on the new dataset(new_appdata10). After feature scaling the data, Logistic Regression(l1 regularization) was used for classification. Initial accuracy of 76.8% was achieved, which was cross-validated using kfold. After this, GridSearch was used, for parameter tuning, and the tuned model gave an accuracy of 76.95% which was an increase, but not a significant one.
#### In the end, the user id, the original enrolment information, and the predicted information were combined.

## Conclusion
#### The model tells us which user will subscribe and who won't. The company can send special offers like 50% of yearly subscription or something else to convert the users whom the model classified as won’t enroll to get them enrolled. This will be profitable for the company. The model can even be implemented daily, and instant offers can be sent as well.
