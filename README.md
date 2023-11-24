# sms_messages_as_spam_or_ham
Use an AutoML tool or library to develop a machine learning model on a given dataset.

## Preprocessing:
- The dataset I used was just a label(`spam` or `ham`) and the message. Due to only having message as a feature to train the model. I had to create new feature with the message feature like which were Message_Length, URL_Count, Uppercase_Count, etc. Then after making these new feature I a pearson correlation to see which features I should use or not use when training my module. Then I split the data into two sets training and testing.

## AutoML:
- I use h2o as my autoML tool of choice, mostly due to it being very ease to use and wide Range of Algorithms.
- split data into training and testing sets
- use training data to train the model using h2o autoML library
- validate the performance of the model using the testing data

## results:
- The model exhibits exceptional predictive accuracy, as evidenced by its low Mean Squared Error (MSE) of 0.01637 and Root Mean Squared Error (RMSE) of 0.12796, indicating precise predictions. The LogLoss value at 0.07033 suggests high confidence in these predictions, and the Mean Per-Class Error of 0.05704 demonstrates balanced performance across classes.

Its discriminatory power is further highlighted by an Area Under the ROC Curve (AUC) of 0.9891 and an Area Under the Precision-Recall Curve (AUCPR) of 0.9676, both exceptionally high values. This implies a superior ability to differentiate between classes and accurately identify positive cases.

The Gini coefficient at 0.9782 reinforces the model's strong discriminatory capacity. The Confusion Matrix shows a high rate of correct classifications with 966 true negatives and 132 true positives, and minimal false negatives and zero false positives, emphasizing the model's precision and reliability.

Overall, the model demonstrates a remarkable balance of accuracy, precision, and class differentiation, making it highly effective for its intended application.

## process:
- I would get data and create new features with existing data
- 
## Install python libraries:
- run make command `make install`

## Run program:
- run make command `make run`
