# Hotel Reservation Cancellation Prediction
This project implements a machine learning solution to predict hotel reservation cancellations based on various features. The model is trained using a dataset and can be accessed via a Flask-based web application that allows users to input their reservation details and receive predictions.

# Table of Contents
Project Overview
Dataset
Model Training
Flask Web Application
Usage
# Project Overview
This project aims to predict whether a hotel reservation will be canceled or not. It uses various machine learning techniques, including:

Decision Tree Classifier
Random Forest Classifier
The project includes data preprocessing, feature selection, hyperparameter tuning, and model evaluation. The model now expects only the top 8 features for input, enhancing the usability and performance of the prediction.

# Dataset
The dataset used for training the models is first_inten_project.csv, which includes various features related to hotel reservations, such as lead time, average price, special requests, number of adults and children, and booking method (online or onsite).

# Model Training
The model training process involves several key steps:

Import Necessary Libraries: Required libraries for data manipulation, visualization, and machine learning are imported.

Load the Dataset: The dataset is loaded using pandas.

Data Preprocessing: This includes handling missing values, encoding categorical features, and dropping irrelevant columns.

Feature Selection: SelectKBest is utilized to select the top 8 features based on their ANOVA F-values, optimizing the model’s performance by focusing on the most significant predictors.

Model Training and Hyperparameter Tuning: Both the Decision Tree Classifier and Random Forest Classifier are trained using GridSearchCV to identify the best hyperparameters for each model.

Model Evaluation: The accuracy, confusion matrix, and classification report are generated for both models, providing insight into their performance.

Save the Models: The trained models are saved using the pickle library for future use.

# Flask Web Application
A Flask web application allows users to input their reservation details and receive predictions about cancellation.

# Key Components of the Flask App:
Create the Application Object: The Flask app is instantiated, and the trained model is loaded.

HTML Template: The HTML template is defined directly in the code for user input, providing fields for reservation details.

Home Route: The home route serves the HTML template, allowing users to interact with the application.

Predict Route: When the user submits the form, the predict route processes the input data, prepares it for the model, and makes a prediction regarding cancellation. Emojis are included in the output for a more engaging user experience.

Running the App: The Flask app runs in debug mode, enabling real-time changes and updates.

# Usage
Train the models using the provided Python script. The script preprocesses the dataset, performs feature selection, and trains both the Decision Tree and Random Forest models.

After training, the best models are saved as best_dt_model.pkl and best_rf_model.pkl.

# Input Features
The application expects the following input features:

Lead Time: Number of days between the booking date and the arrival date.
Average Price: The average price of the stay.
Number of Special Requests: Number of special requests made by the guest.
Reservation Year: The year of the reservation.
Number of Adults: Total number of adults in the reservation.
Number of Children: Total number of children in the reservation.
Booking Method: Indicates whether the booking was made online (1) or onsite (0).
# Output
Upon submission of the input form, the application will display the predicted reservation status with emojis:

Canceled 😞: Indicates the reservation is predicted to be canceled.
Not Canceled 😊: Indicates the reservation is predicted to be not canceled.
# Visualizations
The script generates visualizations such as confusion matrices to evaluate model performance and provides classification reports detailing the precision, recall, and F1-score.
