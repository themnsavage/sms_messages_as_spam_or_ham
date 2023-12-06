# sms_messages_as_spam_or_ham
Use an AutoML tool or library to develop a machine learning model on a given dataset.

## Preprocessing:
The dataset I used was just a label(`spam` or `ham`) and the message. Due to only having message as a feature to train the model. I had to create new feature with the message feature like which were Message_Length, URL_Count, Uppercase_Count, etc. Then after making these new feature I a pearson correlation to see which features I should use or not use when training my module. Then I split the data into two sets training and testing.

## AutoML:
- I use h2o as my autoML tool of choice, mostly due to it being very ease to use and wide Range of Algorithms.
- split data into training and testing sets
- use training data to train the model using h2o autoML library
- validate the performance of the model using the testing data

## results:
The model exhibits exceptional predictive accuracy, as evidenced by its low Mean Squared Error (MSE) of 0.01637 and Root Mean Squared Error (RMSE) of 0.12796, indicating precise predictions. The LogLoss value at 0.07033 suggests high confidence in these predictions, and the Mean Per-Class Error of 0.05704 demonstrates balanced performance across classes.

Its discriminatory power is further highlighted by an Area Under the ROC Curve (AUC) of 0.9891 and an Area Under the Precision-Recall Curve (AUCPR) of 0.9676, both exceptionally high values. This implies a superior ability to differentiate between classes and accurately identify positive cases.

The Gini coefficient at 0.9782 reinforces the model's strong discriminatory capacity. The Confusion Matrix shows a high rate of correct classifications with 966 true negatives and 132 true positives, and minimal false negatives and zero false positives, emphasizing the model's precision and reliability.

Overall, the model demonstrates a remarkable balance of accuracy, precision, and class differentiation, making it highly effective for its intended application.

## process:
- Get data set and create new features with existing data
- Split data into training and testing set
- pearson correlation on feature to see which one to use
- train model with training data
- validate performace of model with testing data
- save model for later use
  
## Install python libraries:
- run make command `make install`

## Run program:
- run make command `make run`

# Technical Report

## Introduction:
-  For this project I will be using the AutoML tool from the python library h2o to create a model to detect if a sms message is spam or ham

## Dataset Selection & Preprocessing:
- The dataset I used just contain a label(spam or ham) and the message. Due to only having message as a feature to train the model I decided to generate new features with the message feature (ex: Message_Length, URL_Count, Uppercase_Count, etc). Then after making these new feature I use a pearson correlation to see which features I should use or not use when training my module. Then I split the data into two sets training and testing.

- pearson correlation: 

## AutoML Implementation:
- The AutoML tool I used was the h2o python library. Using this tool I first train a model using my training data set. After this I would then evaluate the performance of the model, using the testing data set. With the evaluation I would make decisions on how to make the ML model better. I also would save the model using the h2o library for later use.

## Analysis:
- One of the limitation of using an AutoML tool is that it's hard to understand what exactly is happening due to a lot of the process being automated by the h2o tool. While if I created the code for the ML model myself. I feel like I would know more about what exactly is going on instead of it being a black box.

## Lessons Learned:
- From this project I learn that you need to generate new features if not given a lot from data set to train your model. At first I tried to develope a model only using the given feature message, but this resulted in a very poor model.

## results:
The model exhibits exceptional predictive accuracy, as evidenced by its low Mean Squared Error (MSE) of 0.01637 and Root Mean Squared Error (RMSE) of 0.12796, indicating precise predictions. The LogLoss value at 0.07033 suggests high confidence in these predictions, and the Mean Per-Class Error of 0.05704 demonstrates balanced performance across classes.

Its discriminatory power is further highlighted by an Area Under the ROC Curve (AUC) of 0.9891 and an Area Under the Precision-Recall Curve (AUCPR) of 0.9676, both exceptionally high values. This implies a superior ability to differentiate between classes and accurately identify positive cases.

The Gini coefficient at 0.9782 reinforces the model's strong discriminatory capacity. The Confusion Matrix shows a high rate of correct classifications with 966 true negatives and 132 true positives, and minimal false negatives and zero false positives, emphasizing the model's precision and reliability.

Overall, the model demonstrates a remarkable balance of accuracy, precision, and class differentiation, making it highly effective for its intended application.

## References:
- data set used: https://archive.ics.uci.edu/dataset/228/sms+spam+collection