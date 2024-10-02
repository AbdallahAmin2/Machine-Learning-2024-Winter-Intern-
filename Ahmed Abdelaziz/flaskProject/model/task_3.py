import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import optuna
from sklearn.feature_selection import RFE, SelectKBest, chi2,f_classif
import joblib
from sklearn.model_selection import train_test_split

def to_date(date):
    try:
        # Convert the cleaned date to datetime with the correct format
        return pd.to_datetime(date, format='%m/%d/%Y')
    except ValueError as e:
        # Handle date conversion error
        print(f"Error: {e}")
        return None  # Return None for invalid dates


df = pd.read_csv("C:/Users/Ahmed/Downloads/first inten project.csv")


df['Booking_ID'] = df['Booking_ID'].fillna('').astype(str)
df['type of meal'] = df['type of meal'].astype('category')
df['car parking space'] = df['car parking space'].astype(bool)
df['room type'] = df['room type'].astype('category')
df['market segment type'] = df['market segment type'].astype('category')
df['repeated'] = df['repeated'].astype(bool)
df['booking status'] = df['booking status'].astype('category')



# Correct invalid dates in the 'date of reservation' column
df.loc[df['date of reservation'] == '2018-2-29', 'date of reservation'] = '2/28/2018'


# Convert the cleaned date to datetime
df['date of reservation'] = df['date of reservation'].apply(to_date)

df_encoding = df.copy()
df_encoding.drop(columns=['Booking_ID'], axis=1, inplace=True)
# Total nights (weekend + weekdays)
df_encoding['total_nights'] = df_encoding['number of weekend nights'] + df_encoding['number of week nights']

# Group size (total number of people)
df_encoding['total_people'] = df_encoding['number of adults'] + df_encoding['number of children']
# Perform One-Hot Encoding on categorical features
df_encoding = pd.get_dummies(df_encoding, columns=['type of meal', 'room type', 'market segment type'], drop_first=True)
categorical_columns = df_encoding.select_dtypes(exclude=['int64' , 'float']).columns

# Extracting date-related features
df_encoding['reservation_year'] = df_encoding['date of reservation'].dt.year
df_encoding['reservation_month'] = df_encoding['date of reservation'].dt.month
df_encoding['reservation_day'] = df_encoding['date of reservation'].dt.day
df_encoding['reservation_weekday'] = df_encoding['date of reservation'].dt.weekday  # 0=Monday, 6=Sunday
df_encoding.drop(columns=['date of reservation'], inplace=True)

non_numeric_columns = df_encoding.select_dtypes(exclude=['int64', 'float64']).columns


# Separate features (X) and target (y)
X = df_encoding.drop(columns=['booking status'])
y = df_encoding['booking status']


# Split the data (e.g., 80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Note: This is a binary classification problem where the target is the 'booking status'.
# Target classes:
# - 0: 'Not_Canceled'
# - 1: 'Canceled'

# Step 1: Scale the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 3: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 4: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Step 5: Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Display the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Assuming y_true is the true labels and y_pred is the model predictions
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

# Show the plot
plt.show()

# Step 3: Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    C = trial.suggest_loguniform('C', 0.01, 100)  # Regularization strength
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 300)  # Maximum number of iterations

    # Initialize the Logistic Regression model with the suggested hyperparameters
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Return the accuracy
    return accuracy_score(y_test, y_pred)

# Step 4: Create a study and optimize
study = optuna.create_study(direction='maximize')  # We want to maximize accuracy
study.optimize(objective, n_trials=50)  # Number of trials

# Step 5: Print the best parameters and the corresponding score
print(f"Best Parameters: {study.best_params}")
print(f"Best Test Accuracy: {study.best_value:.4f}")

# Step 6: Train the model with the best parameters
best_params = study.best_params
best_model = LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=best_params['max_iter'])
best_model.fit(X_train_scaled, y_train)

# Step 7: Make predictions with the best model
y_pred_best = best_model.predict(X_test_scaled)

# Step 8: Evaluate the performance
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test Accuracy with Best Parameters: {accuracy_best:.4f}")

# Step 9: Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Step 10: Display the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Assuming y_true is the true labels and y_pred is the model predictions
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

# Show the plot
plt.show()


# Step 3: RFE for feature selection
logreg = LogisticRegression(max_iter=1000)  # Logistic Regression for RFE
rfe = RFE(estimator=logreg, n_features_to_select=10)  # Select top 10 features
rfe.fit(X_train_scaled, y_train)

# Step 4: Get the selected features using original column names
selected_features_rfe = X.columns[rfe.support_]  # Get column names using boolean mask
print(f"Selected Features from RFE: {selected_features_rfe}")

# Transform the training and test data using only the selected features
X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)


# ------------------- Feature Selection using SelectKBest -------------------
# Step 5: SelectKBest for feature selection
select_k_best = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features based on ANOVA F-value
select_k_best.fit(X_train_scaled, y_train)

# Step 6: Get the selected features from SelectKBest
selected_features_kbest = X.columns[select_k_best.get_support()]  # Get selected feature names
print(f"Selected Features from SelectKBest: {selected_features_kbest}")

# Transform the training and test data using only the selected features
X_train_kbest = select_k_best.transform(X_train_scaled)
X_test_kbest = select_k_best.transform(X_test_scaled)

# ------------------- Train and Evaluate Logistic Regression Model -------------------
model_rfe = LogisticRegression(max_iter=1000)
model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)

# Step 8: Evaluate the model using RFE-selected features
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f"Accuracy after RFE: {accuracy_rfe:.4f}")
print("\nClassification Report (RFE):")
print(classification_report(y_test, y_pred_rfe))

# Step 9: Display confusion matrix for RFE
cm_rfe = confusion_matrix(y_test, y_pred_rfe)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rfe, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
plt.title('Confusion Matrix for Logistic Regression after RFE')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 10: Train the model using SelectKBest-selected features
model_kbest = LogisticRegression(max_iter=1000)
model_kbest.fit(X_train_kbest, y_train)
y_pred_kbest = model_kbest.predict(X_test_kbest)

# Step 11: Evaluate the model using SelectKBest-selected features
accuracy_kbest = accuracy_score(y_test, y_pred_kbest)
print(f"Accuracy after SelectKBest: {accuracy_kbest:.4f}")
print("\nClassification Report (SelectKBest):")
print(classification_report(y_test, y_pred_kbest))

# Step 12: Display confusion matrix for SelectKBest
cm_kbest = confusion_matrix(y_test, y_pred_kbest)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_kbest, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
plt.title('Confusion Matrix for Logistic Regression after SelectKBest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Save the trained model to a file using Joblib
joblib.dump(model_rfe, 'logistic_regression_model.joblib')

# Load the model back from the file
loaded_model = joblib.load('logistic_regression_model.joblib')

# Make predictions using the loaded model
y_pred_loaded = loaded_model.predict(X_train_rfe)

