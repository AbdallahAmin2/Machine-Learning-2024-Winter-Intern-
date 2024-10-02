from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Predict
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        Final_Prediction = "Cancelled"
    elif  prediction[0] == 1:
        Final_Prediction = "Not Cancelled"

    
    # Prepare the result message
    result = f'The predicted class is: {Final_Prediction}'
    
    return render_template('index.html', prediction_text=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
