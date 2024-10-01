from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load the trained model and scaler
def load_model():
    model_path = 'rf_model.pkl'
    scaler_path = 'scaler.pkl'
    
    with open(model_path, 'rb') as model_file:
        rf_model = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    return rf_model, scaler

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('frontend.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    rf_model, scaler = load_model()

    # Get the form data
    input_data = [
        request.form['car_parking_space'],
        request.form['lead_time'],
        request.form['repeated'],
        request.form['average_price'],
        request.form['special_requests'],
        request.form['year'],
        request.form['market_segment_type_Complementary'],
        request.form['market_segment_type_Corporate'],
        request.form['market_segment_type_Offline'],
        request.form['market_segment_type_Online']
    ]

    # Convert input to numpy array and reshape for the model
    input_data = np.array([float(i) for i in input_data]).reshape(1, -1)

    # Scale the features
    scaled_features = scaler.transform(input_data)

    # Predict using the model
    prediction = rf_model.predict(scaled_features)
    
    if int(prediction[0]) == 1:
        result = "not canceled."
    else:
        result = "canceled."

    # Render the result
    return render_template('frontend.html', prediction=f'The predicted booking state is: {result}')
if __name__ == '__main__':
    app.run(debug=True)