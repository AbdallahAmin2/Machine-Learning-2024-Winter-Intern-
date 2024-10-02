import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Load the pre-trained Logistic Regression model
loaded_model = joblib.load('model/logistic_regression_model.joblib')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and convert to the appropriate types
        number_of_week_nights = float(request.form['number_of_week_nights'])
        lead_time = float(request.form['lead_time'])
        repeated = bool(int(request.form['repeated']))  # Convert to boolean
        average_price = float(request.form['average_price'])

        # Handle special requests; if it's a count, convert to int
        special_requests = int(request.form['special_requests'])  # Change to int if it's a count

        total_nights = float(request.form['total_nights'])
        total_people = float(request.form['total_people'])

        # Convert market segment types to binary
        market_segment_type_corporate = 1 if request.form['market_segment_type'] == 'Corporate' else 0
        market_segment_type_online = 1 if request.form['market_segment_type'] == 'Online' else 0

        reservation_year = int(request.form['reservation_year'])

        # Create an array for the model input
        features = np.array([
            number_of_week_nights,
            lead_time,
            repeated,
            average_price,
            special_requests,  # This should now be a numeric value
            total_nights,
            total_people,
            market_segment_type_corporate,
            market_segment_type_online,
            reservation_year
        ]).reshape(1, -1)

        # Make predictions
        prediction = loaded_model.predict(features)
        # Return the prediction result
        print(prediction[0])
        if prediction[0] == 'Canceled':
            result = 'Canceled'
        else:
            result = 'Not Canceled'

        return render_template('index.html', prediction=result)

    except ValueError as ve:
        print(f"Value error occurred: {ve}")
        return render_template('index.html', prediction='Invalid input. Please check your values.')
    except Exception as e:
        print(f"Error occurred: {e}")  # Log the error to the console
        return render_template('index.html', prediction='Error occurred while processing your request.')


if __name__ == '__main__':
    app.run(debug=True)
