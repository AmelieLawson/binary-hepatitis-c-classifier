# Hepatitis C Classifier

My first machine learning project using Python and scikit-learn.

## Project Overview

The aim of this project was to use machine learning to classify individuals based on their age, sex, and various laboratory values.

The original dataset labelled individuals as either blood donors, suspected blood donors, hepatitis C, fibrosis, or cirrhosis. This project was simplified to a binary classification problem where individuals were classified as either blood donors (blood donors and suspected blood donors) or hepatitis C patients (hepatitis C, fibrosis and cirrhosis).

The data was used to train a Random Forest Classifier model to predict whether an unknown individual was a blood donor or hepatitis C patient.

## Dataset

The dataset used in this project is the Hepatitis C Prediction Dataset, obtained from Kaggle. It is not included in this repository. Place `HepatitisCdata.csv` in the same directory as `hepatitis.py` before running the program.

[Hepatitis C Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset)

The data was originally obtained from UCI Machine Learning Repository. It contains laboratory values for a range of blood donors and hepatitis C patients along with demographic values like age and sex.


## Method

The steps involved in the project are outlined below:

1. Load the dataset using pandas.
2. Convert "Sex" variable from categoric (m/f) to numerical (0/1).
3. Convert the original five disease categories into a binary target variable.
4. Split the data into a training set and a test set using a 75/25 split.
5. Train a Random Forest Classifier with 100 estimators.
6. Predict the outcomes of the test set.
7. Evaluate the model using its accuracy, ROC-AUC, and classification report.
8. Allow the user to enter data for a new individual and generate a prediction.

## Evaluation

The metrics used the evaluate the model were as follows:

- Accuracy: The proportion of test predictions which were correct.
- ROC/AUC: A quantification of how well the model distinguished between the classes (i.e. the balance between sensitivity and rate of false positives)
- Classification report: Includes precision (the proportion of positive predictions which were accurate), recall (the proportion of positive cases that the model managed to find), and F1 score (precision and recall combined into a single metric) for each class.

## How to Run

Install the required Python libraries:

```bash
pip install pandas scikit-learn
```

## What I Learnt

Below is a summary of what I have learnt from my first machine learning project:

- How to load a dataset into a Python file.
- How to convert a categoric variable into a numeric one.
- How to define the features and the target variable.
- How to split data into a training and a test set.
- How to train a Random Forest Classifier model.
- What each evaluation metric measures and how they differ.
- How to use a machine learning model to generate a prediction based on new data.

## Opportunities for Advancement

I could try to build a classifier which retains all five original categories, instead of simplifying to a binary classification problem.
