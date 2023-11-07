#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Parameters
output_file = "random_forest_model.pkl"


# load data
print("load data")
path="data/heart.csv"
df=pd.read_csv(path)
df.head()
print()
# DATA PREPARATION
print("data preparation")

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Split the dataset into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

vectorizer = DictVectorizer(sparse=False)

X_train = vectorizer.fit_transform(X_train.to_dict(orient='records'))
X_val = vectorizer.transform(X_val.to_dict(orient='records'))
X_test = vectorizer.transform(X_test.to_dict(orient='records'))
print()
# MODELLING
print("modelling")
print()
# Define and initialize the logistic regression model
log_model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)

# Fit the model to the training data
log_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = log_model.predict(X_train)
y_val_pred = log_model.predict(X_val)

# Calculate accuracy for training and test data
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_val, y_val_pred)
print("Train logistic regression model")
print("Training Accuracy:", round(acc_train, 2))
print("Test Accuracy:", round(acc_test, 2))
print()



print("Tune random forest model")
# Initialize variables to store the best parameters and the highest AUC score
best_max_depth = 0
best_n_estimators = 0
best_auc = 0

# Define a range of values for max_depth and n_estimators to search
max_depth_values = [5, 10, 15, 20]
n_estimators_values = [10, 20, 30, 40]

# Iterate over max_depth and n_estimators combinations
for max_depth in max_depth_values:
    for n_estimators in n_estimators_values:
        # Create a RandomForestClassifier  model with the current parameters
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1)
        # Train the model on the training data
        model.fit(X_train, y_train)
        # Make predictions on the validation data
        y_pred_proba = model.predict_proba(X_val)[:, 1] 
        # Calculate the AUC score
        auc = roc_auc_score(y_val, y_pred_proba)
        # Check if the current AUC score is the best
        if auc > best_auc:
            best_auc = auc
            best_max_depth = max_depth
            best_n_estimators = n_estimators

# Print the best parameters and the highest AUC score
print(f"Best max_depth: {best_max_depth}")
print(f"Best n_estimators: {best_n_estimators}")
print(f"Highest AUC score: {best_auc}")
print()


print("training random forest model")
# Create a Random Forest model with the best parameters
rf_model = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_n_estimators, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)

# Calculate accuracy for training and test data
acc_train = accuracy_score(y_train, y_train_pred)
acc_val = accuracy_score(y_val, y_val_pred)

print("Training Accuracy:", round(acc_train, 2))
print("Validation Accuracy:", round(acc_val, 2))

# Evaluate the best model on the test set
y_val_pred_proba = rf_model.predict_proba(X_val)[:, 1]
test_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"AUC score on validation data: {test_auc:.2f}")
print()




# In[27]:

print("tune xgboost model")
import xgboost as xgb
from xgboost import DMatrix, train

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

best_params = None
best_auc = 0

for max_depth in param_grid['max_depth']:
    for learning_rate in param_grid['learning_rate']:
        # Create a DMatrix for training data
        dtrain = DMatrix(X_train, label=y_train)
        # Create a DMatrix for testing data
        dtest = DMatrix(X_val, label=y_val)

        # Set hyperparameters
        params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'nthread': 8,
            'seed': 42
        }

        # Train the XGBoost model
        model = train(params, dtrain, num_boost_round=100)
        # Make predictions on the validation data
        y_pred_proba = model.predict(dtest)

        # Calculate the AUC score
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_params = (max_depth, learning_rate)

print(f"Best hyperparameters: max_depth={best_params[0]}, learning_rate={best_params[1]}")
print(f"Highest AUC score: {best_auc:.2f}")
print()

print("training xgboost model")
# Create the best XGBoost model
xgb_model = xgb.XGBClassifier(
    max_depth=best_params[0],
    learning_rate=best_params[1],
    random_state=42
)

# Train the best model on the entire dataset
xgb_model.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)

# Calculate accuracy for training and test data
acc_train = accuracy_score(y_train, y_train_pred)
acc_val = accuracy_score(y_val, y_val_pred)
print("Training Accuracy:", round(acc_train, 2))
print("Validation Accuracy:", round(acc_val, 2))

# Evaluate the best model on the test set
y_val_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
test_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"AUC score on test data: {test_auc:.2f}")
print()


# COMPARISON

# In[33]:

print("comparing models")
from prettytable import PrettyTable
# Evaluate models
models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Create a table
results_table = PrettyTable()
results_table.field_names = ["Model", "Accuracy", "AUC"]

# Populate the table with results
for model_name, model in models.items():
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    results_table.add_row([model_name, f"{accuracy:.2f}", f"{auc:.2f}"])

# Print the table
print(results_table)


print("Based on the accuracy and AUC scores, it appears that the Random Forest models is performing the best,\
     with both having high accuracy and AUC scores.")
print()
import pickle

# Save the model to a file
with open( output_file, 'wb') as file:
    model_data = {
        'model': rf_model,
        'vectorizer': vectorizer
    }
    pickle.dump(model_data, file)
print("model and vectorizer successfully saved")