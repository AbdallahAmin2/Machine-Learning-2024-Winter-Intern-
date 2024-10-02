from flask import Flask, render_template_string, request
import pickle
import numpy as np

# Create the application object
app = Flask(__name__)

# Load the Random Forest model
try:
    model = pickle.load(open('best_rf_model.pkl', 'rb'))  # Load the Random Forest model
except Exception as e:
    print(f"Error loading the model: {e}")

# Combined HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>Hotel Reservation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <link rel="preconnect" href="https://fonts.googleapis.com"/>
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
    <link
            href="https://fonts.googleapis.com/css2?family=Montserrat:wght@100;200;300;400;500;600;700;800;900&display=swap"
            rel="stylesheet"
    />
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"/>
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
    <script src="https://kit.fontawesome.com/6d56012fea.js" crossorigin="anonymous"></script>
    <style>
        body {
            background-color: #E6E6FA; /* Lavender background */
        }
        .container {
            font-family: "Montserrat", sans-serif;
            margin-top: 30px;
            display: flex;
            justify-content: space-between;
        }
        h1 {
            font-weight: 900;
            font-size: 40px;
            margin-bottom: 30px;
            color: #8B008B; /* Purple color for the heading */
        }
        .form-section {
            width: 70%;
        }
        .result-section {
            width: 20%;
        }
        i {
            color: 	#663399; /* Purple color for icons */
        }
        input[type="number"] {
            width: 90%;
            border-radius: 10px;
            -moz-appearance: textfield;
        }
        #ans {
            font-weight: 700;
            font-size: 28px;
            color: #562A56; /* Purple color for the result */
        }
        .result {
            font-size: larger;
            color:#8B008B;
        }
        .result-content p {
            margin-top: 30px;
        }
        label {
            font-weight: 500;
            font-size: large;
            color: #8B008B; /* Purple color for labels */
        }
        input[type="radio"] {
            margin-right: 20px;
        }
        .btn-primary {
            margin-top: 20px;
            font-weight: 700;
            background-color:#8B008B; /* Purple color for the submit button */
            border: none; /* Remove default border */
        }
        .btn-primary:hover {
            background-color: 	#DA70D6; /* Darker purple on hover */
        }
    </style>
</head>
<body>
<div class="container">
    <form action="/predict" class="form-group form-section" method="post">
        <h1>Hotel Reservation Cancellation Prediction <i class="fa-solid fa-hotel"></i></h1>
        <div class="mb-3">
            <label for="lead_time" class="form-label"><strong>Lead Time:</strong></label>
            <input type="number" class="form-control" id="lead_time" required placeholder="Enter Lead Time" name="lead_time"/>
        </div>
        <hr>
        <div>
            <label for="avg_price" class="form-label"><strong>Average Price:</strong></label>
            <input type="number" class="form-control" id="avg_price" required placeholder="Enter Average Price" name="avg_price"/>
        </div>
        <hr>
        <div class="mb-3">
            <label for="special_request" class="form-label"><strong>Number of Special Requests:</strong></label>
            <input type="number" class="form-control" id="special_request" required placeholder="Enter Number of Special Requests" name="special_request"/>
        </div>
        <hr>
        <div class="mb-3">
            <label for="year" class="form-label"><strong>Reservation Year:</strong></label>
            <input type="number" class="form-control" id="year" required placeholder="Enter Reservation Year" name="year"/>
        </div>
        <hr>
        <div class="mb-3">
            <label for="adults" class="form-label"><strong>Number of Adults:</strong></label>
            <input type="number" class="form-control" id="adults" required placeholder="Enter Number of Adults" name="adults"/>
        </div>
        <hr>
        <div class="mb-3">
            <label for="children" class="form-label"><strong>Number of Children:</strong></label>
            <input type="number" class="form-control" id="children" required placeholder="Enter Number of Children" name="children"/>
        </div>
        <hr>
        <div>
            <label class="form-label"><strong>Booking Method:</strong> </label>
            <div class="form-check">
                <input class="form-check-input" type="radio" name="method" id="online" value="1">
                <label class="form-check-label" for="online">Online</label>
            </div>
            <div class="form-check">
                <input class="form-check-input" type="radio" name="method" id="onsite" value="0" checked>
                <label class="form-check-label" for="onsite">Onsite</label>
            </div>
        </div>
        <hr>
        <button type="submit" class="btn btn-primary">Submit</button>
    </form>
    <div class="result result-section">
        <h2>Result <i class="fa-solid fa-scroll"></i></h2>
        <div class="result-content">
            <p>Reservation: <br><span id="ans">{{ prediction }}</span></p>
        </div>
    </div>
</div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    lead_time = int(request.form['lead_time'])
    avg_price = float(request.form['avg_price'])
    special_request = int(request.form['special_request'])
    year = int(request.form['year'])
    adults = int(request.form['adults'])
    children = int(request.form['children'])
    method = int(request.form['method'])  # Online (1) or Onsite (0)

    # Prepare features for prediction
    # Adjust to ensure you have 24 features as expected by the model.
    features = np.array([lead_time, avg_price, special_request, year, adults, children, method, 0]).reshape(1, -1)  # Add necessary features

    # Make prediction
    pred = model.predict(features)
    
    # Add emojis based on the prediction
    if pred[0] == 1:
        prediction = "Canceled 😞"  # Frowning face for canceled
    else:
        prediction = "Not Canceled 😊"  # Smiling face for not canceled

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)







