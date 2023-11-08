# EarlyHeart: Predictive Modeling for Early Heart Disease Detection

![HeartDisease](images/Heart-disease-know-your-risk.jpg)

## Project Overview

The "Heart Disease Prediction" project aims to develop a predictive model that can assist healthcare professionals in assessing the risk of heart disease in patients. Heart disease remains a leading cause of mortality worldwide, and early diagnosis and intervention are crucial for improving patient outcomes. This project leverages data-driven approaches to aid medical practitioners in making informed decisions.

## Problem Statement

Heart disease is a complex condition influenced by various factors, including demographic, clinical, and lifestyle-related features. The goal of this project is to build a robust predictive model that can:
- Accurately predict the likelihood of heart disease based on patient data.
- Provide actionable insights to medical professionals for early diagnosis and timely intervention.

## Target Audience

The primary stakeholders and target audience for this project include:
- Cardiologists and medical practitioners: They can use the predictive model to assess patients' heart disease risk, allowing for proactive measures and personalized treatment plans.
- Patients: Individuals can benefit from risk assessments, encouraging health-conscious decisions and early consultation with healthcare providers.

## Value Proposition

The "Heart Disease Prediction" project offers several key value propositions:

1. **Early Detection**: The model enables early detection of heart disease, which is crucial for timely intervention and better patient outcomes.

2. **Personalized Healthcare**: Medical professionals can tailor treatment plans based on individual risk assessments, ensuring more effective and patient-centric care.

3. **Resource Optimization**: Healthcare facilities can allocate resources more efficiently by focusing on high-risk patients, reducing unnecessary tests and costs.

4. **Patient Empowerment**: Patients gain insights into their own health and can make informed decisions to mitigate risk factors.

5. **Reduced Mortality**: The project's success can contribute to reduced heart disease-related mortality by promoting early diagnosis and appropriate interventions.

## Project Goals

The overarching goals of the "Heart Disease Prediction" project are as follows:

1. Develop a robust predictive model for heart disease that achieves high accuracy and reliability.
2. Deploy the model as an API endpoint, allowing for real-time predictions.
3. Facilitate collaboration between data scientists and healthcare professionals to maximize the model's clinical utility.
4. Promote early diagnosis, patient engagement, and personalized treatment strategies to improve patient outcomes and reduce the burden of heart disease.

## Project Success Criteria

The project's success will be assessed based on the following criteria:

1. Model Performance: The model should achieve a high level of accuracy and AUC score when predicting heart disease risk.
2. Clinical Utility: The model should be easy to integrate into clinical practice and provide actionable insights.
3. Adoption Rate: The successful adoption of the predictive model by healthcare professionals and patients.
4. Reduction in Mortality: A reduction in heart disease-related mortality through early diagnosis and interventions.

This "Heart Disease Prediction" project aligns with the overarching goal of improving healthcare outcomes and addressing a critical global health concern.


## Data Analysis

We began by exploring the dataset, which contained information on patients' age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved during exercise, exercise-induced angina, ST depression induced by exercise relative to rest (Oldpeak), and the slope of the peak exercise ST segment (ST_Slope). We also had a target variable, 'HeartDisease,' which indicated the presence or absence of heart disease.

Key observations from the data analysis include:
- The dataset contains both numerical and categorical features.
- The dataset has 918 samples with a balanced distribution of positive and negative heart disease cases.

## Model Building and Evaluation

We built a predictive model using a Random Forest classifier to predict the likelihood of heart disease in patients. The model achieved promising results in terms of accuracy and AUC score on the validation set.

Key highlights of the model-building process:
- The model was trained and validated using a combination of numerical and categorical features.
- Feature importance analysis revealed the most critical predictors of heart disease, with 'ST_Slope=Up' being the most influential feature.
- We also evaluated the model using the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) to assess its performance.

## Project Structure

- `/data`: This directory contains the dataset used for the project.

1. `notebook.ipynb`: This Jupyter notebook contains the data exploration, analysis, and visualization steps of my project. It's where I developed and tested my machine learning model and data processing code. You can refer to this notebook for a detailed walkthrough of my project's development process.

2. `Dockerfile`: This file contains the configuration for building a Docker container for my project. Docker allows me to create a consistent and isolated environment for my application. This file defines the necessary packages and dependencies required to run my project within a container.

3. `train.py`: This Python script is responsible for training my machine learning model. It loads data from a dataset, preprocess it, and then fits the model on the training data. Running this script will train my model and generate the necessary model files or artifacts.

4. `predict.py`: This Python script serves as the application for making predictions with my trained model. It loads the trained model, processes incoming data, and returns predictions. It can be used for batch predictions or integrated into a web application for real-time predictions.

5. `Pipfile`: This file lists the Python packages and dependencies required for my project. It's used with the Pipenv tool to manage my project's virtual environment and package versions.

6. `Pipfile.lock`: This is a generated file by Pipenv that specifies the exact versions of packages and dependencies used in my project. It ensures that my project uses the same package versions across different environments.

7. `random_forest_model.pkl`: This is the saved machine learning model that I trained. You can use this model to make predictions without retraining it every time. This file can be loaded using libraries like `pickle` in Python.

8. `test_prediction.py`: This Python script tests my prediction functionality. It sends sample data to my prediction endpoint (if applicable) and checks if it returns the expected results. It's used to validate the prediction system.

## Deployment

We deployed the trained Random Forest model as a predictive API endpoint using Flask. This API can accept patient data as input and provide predictions regarding the presence of heart disease. The deployment allows for real-time prediction of heart disease risk for new patients.

## Usage:

1. To train my machine learning model, run `train.py`. This script will load and preprocess my data, train the model, and save it to the `random_forest_model.pkl` file.

2. To make predictions using my trained model, run `predict.py`. You can input data, and it will use the trained model to provide predictions.

3. If you want to test my prediction system, run `test_prediction.py`. This script sends sample data to my prediction endpoint (if deployed), and it checks if the predictions match the expected results.

4. If you have Docker installed, you can build and run my project within a Docker container using the `Dockerfile`. This is especially useful for creating an isolated and reproducible environment for the project.

Make sure to have the required dependencies and packages specified in the `Pipfile` and `Pipfile.lock` installed in your Python environment before running any scripts.

## Deployment on AWS Elastic Beanstalk

This project is deployed on AWS Elastic Beanstalk. You can access it at the following URL:

[AWS Elastic Beanstalk URL](http://heart-serving-env.eba-pahrexjp.eu-west-3.elasticbeanstalk.com/)

Feel free to reach out if you have any questions or need further assistance!


## Future Improvements

In future iterations of this project, the following enhancements can be considered:
- Collecting additional data to increase the size and diversity of the dataset, potentially improving model performance.
- Fine-tuning the model parameters to achieve even better predictive accuracy.
- Implementing additional features, such as handling missing data and dealing with class imbalances, to further enhance model robustness.

Overall, this project provides a valuable tool for assessing the risk of heart disease in patients, which can be used in a clinical setting for early diagnosis and intervention.

# Contact
Follow me on Twitter 🐦, connect with me on LinkedIn 🔗, and check out my GitHub 🐙. You won't be disappointed!

👉 Twitter: https://twitter.com/NdiranguMuturi1  
👉 LinkedIn: https://www.linkedin.com/in/isaac-muturi-3b6b2b237  
👉 GitHub: https://github.com/Isaac-Ndirangu-Muturi-749  