# Likely hood of E-signing a loan

## Business Problems
#### The provided dataset is for the customer base of a bank. The bank has tie-ups with a P2P company who send them potential leads about people who are interested in taking a loan. The goal of this project was to determine if a person will sign go to the final page of signing an E-contract for a loan or no. If a person reaches that page, he/she is said to be a quality applicant. Thus, a model needed to be designed which would predict if an applicant is a quality applicant or no.

## Model
#### In the EDA phase, a histogram, a correlation plot, and a heat map were plotted to visualize and explore the data. This exploration showed that the ‘months_employed’ column had a large number of 0 values. This could affect our model. Hence, this column was dropped during the feature engineering phase.
#### This was followed by encoding the categorical data and feature scaling.
#### Various models were used to see which one gave the best accuracy. The table below shows the accuracy’s for various models.

| Model | Accuracy |
| --- | --- |
| Logistic Regression (l1 regularization) | 56.2% |
| SVM |56.8% |
| Kernel SVM | 59.2% |
| Random Forest (n=100) | 62.2% |

#### Thus, from the accuracies, it was clear that Random Forest was the best model. K-fold cross-validation validated the accuracy. This was followed by using GridSearch for parameter tuning. The model with tuned parameters gave an accuracy of 63.53%.
#### This tuned model was then used to build the confusion matrix, followed by creating a data frame combining the user id, actual signing data, and predicted data.

## Conclusion
#### Thus, the model shows that around 40% of the people who come to the bank from the P2P company do not convert. These customers can be converted, as well, by designing specialized onboarding processes for them depending on the likelihood of them signing a loan. Better rates or some extra perks can be provided in this onboarding phase for a higher conversion rate.
