from flask import Flask, render_template, request
import pickle
import numpy as np
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and the preprocessor
model = pickle.load(open('hotel_booking_model (1).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        lead_time = int(request.form['lead_time'])
        avg_price = float(request.form['avg_price'])
        num_special_requests = int(request.form['num_special_requests'])
        reservation_year = int(request.form['reservation_year'])
        booking_method = request.form['booking_method']

        # Convert booking method to one-hot encoding
        booking_method_online = 1 if booking_method == 'Online' else 0
        booking_method_onsite = 1 if booking_method == 'Onsite' else 0

        # Create a feature vector
        features = np.array([[lead_time, avg_price, num_special_requests,
                              reservation_year, booking_method_online,
                              booking_method_onsite,0,0,0,0]])

        # Predict using the model
        prediction = model.predict(features)

        output = "Canceled" if prediction[0] == 1 else "Not Canceled"
        # Return the result
        return render_template('index.html', prediction=output)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
