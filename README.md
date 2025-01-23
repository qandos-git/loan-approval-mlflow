# loan-approval-mlflow

## Introduction:

This project aims to build a machine learning model to predict loan approval status based on applicant data. The dataset, sourced from Kaggle, contains features such as income, loan amount, credit history, and marital status, which are commonly used to assess eligibility for loans.

## Objectives:

1. Perform exploratory data analysis (EDA) to understand the dataset.

2. Preprocess the data, including encoding categorical variables, and scaling numerical features, standardizing all features.

3. Use cross-validation to get generalized results for hyperparameters tuning.

4. Use pipeline sklearn class for training and evaluation.

4. Train and evaluate classification models to predict loan approval status.

5. Handling imbalanced classes.

5. Optimize the model to achieve a high F1 score, because recall and precision are essential for loan approval.

# Dataset Overview

The dataset contains:

- **Features:** Applicant-related details, such as income, marital status, credit history, etc.

- **Target Variable:** Loan status (Approved or Not Approved).

By analyzing and modeling this data, we aim to create a robust and interpretable solution for loan approval prediction, which could potentially aid financial institutions in making more data-driven decisions

[dataset link](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

## Results
The third column represents the best experiment results
![Results](https://github.com/qandos-git/loan-approval-mlflow/blob/main/artifacts/images/image.png)

After comparing all the experiments, we can see that **RandomForestClassifier with SMOTE** gives the most robust performance, so we will test it.

**Model test results:**

              precision    recall  f1-score   support

           0       0.94      0.96      0.95      6987
           1       0.83      0.79      0.81      1965
    accuracy                           0.92      8952
     macro avg       0.89      0.87      0.88      8952
    weighted avg       0.92      0.92      0.92      8952

As we care about the two classes equally, this performance is very good.
so, we will register the model and use it.


## Challenges
RandomForest classifier is more complex compared to LogisticRegression, but this reflects the problem complexity. Personally, I prefer to use XGBoost classifier, but it seems to conflict with sklearn cross-validate, I have tried it and got `NaN` as F1 output, so resolving this conflict can improve the solution performance.

## Acknowledgment
This Readme file was written with the assistance of ChatGPT
