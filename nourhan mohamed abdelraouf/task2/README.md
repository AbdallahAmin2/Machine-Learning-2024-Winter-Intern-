# Machine-Learning-2024-Winter-Intern-task2
# Data Science Project: Booking Status Prediction
# Overview
This project aims to predict the booking status of hotel reservations using machine learning algorithms. The dataset is processed, cleaned, and analyzed to build models that can accurately classify whether a booking will be successful or not.

# Contents
Data preprocessing
Outlier detection and removal
Feature normalization
Dimensionality reduction using PCA
Model building and evaluation (KNN and Logistic Regression)
Visualization of results
# Technologies Used
Python
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
SciPy
# Dataset
The dataset used in this project is first inten project.csv. It contains various features related to hotel bookings, including:

Booking ID
Date of reservation
Type of meal
Room type
Market segment type
Booking status (the target variable)

# Methodology
Data Loading: The dataset is loaded into a pandas DataFrame.
Data Preprocessing: This includes handling missing values, encoding categorical variables, and feature engineering (date conversion).
Outlier Detection: Outliers are identified and removed using the IQR method.
Train-Validation-Test Split: The dataset is split into training, validation, and test sets.
Feature Scaling: Features are standardized using StandardScaler.
Dimensionality Reduction: PCA is applied to reduce the dimensionality of the feature space while retaining 95% variance.
Model Training: KNN and Logistic Regression models are trained and evaluated on the validation set.
Model Evaluation: Models are assessed using accuracy, confusion matrix, and classification reports.
