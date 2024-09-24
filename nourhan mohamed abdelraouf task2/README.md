# Machine-Learning-2024-Winter-Intern-task2
# Data Preprocessing
The code includes data preprocessing steps such as:

- Checking for null values and data types.
- Removing leading/trailing whitespaces.
- Filling missing values using median (for numeric) and mode (for categorical).
- Outlier Detection
- Outliers are handled using two methods:

# Interquartile Range 
- IQR
- Z-Score
- Both methods are implemented to ensure robust data cleaning.

# Feature Engineering
- A correlation matrix is computed to evaluate relationships between features.
- Categorical features are transformed using Label Encoding to prepare for modeling.
# Modeling
Two machine learning models are implemented:

- Random Forest Classifier
- K-Nearest Neighbors (KNN)
# Model performance is evaluated using accuracy, confusion matrices, and classification reports.

# Results
- Random Forest Accuracy (without dropping):  0.8721094856064181
- KNN Accuracy (without dropping):  0.8357715903728173
- Random Forest Accuracy (with dropping): 0.8681284743668931
- KNN Accuracy (with dropping): 0.8411056207535516


# Output Differences
- Without Dropping Constant Columns: The model may include irrelevant features, which can lead to overfitting or reduced performance.
- With Dropping Constant Columns: The models generally show improved accuracy and robustness by focusing only on informative features.
  # By comparing both versions, users can observe how dropping constant columns impacts model performance.
