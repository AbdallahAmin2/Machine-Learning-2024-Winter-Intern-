from flask import Flask, request, render_template, send_file
import numpy as np
import pandas as pd
import pickle
import os

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        x1 = request.form['lead time']
        x2 = request.form['average price']
        x3 = request.form['special requests']
        x4, x5, x6 = 0, 0, 0
        market_segment = request.form['market_segment']
        if market_segment == '1':
            x4 = 1
        elif market_segment == '2':
            x5 = 1
        elif market_segment == '3':
            x6 = 1

        data = np.array([[x1, x2, x3, x4, x5, x6]], dtype=float)
        prediction = model.predict(data)

        result = 'Booking may be Canceled' if prediction[0] == 1 else 'Booking probably will not be Canceled'
        return render_template('index.html', prediction_result=result)

    return render_template('index.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        if 'myfile' not in request.files:
            return "No file uploaded", 400

        file = request.files['myfile']
        if file.filename == '':
            return "No file selected", 400

        # Read the CSV file
        df = pd.read_csv(file)

        # Assuming the CSV file contains the needed columns
        if not all(col in df.columns for col in ['lead time', 'average price', 'special requests','market segment type_Corporate',
                                                  'market segment type_Offline', 'market segment type_Online']):
            return "Invalid CSV format", 400

        predictions = model.predict(df)

        df['prediction'] = ['Canceled' if pred == 1 else 'Not Canceled' for pred in predictions]

        # # Save the full results for download
        output_file = "predictions.csv"
        df.to_csv(output_file, index=False)

        def map_market_segment(row):
            if row['market segment type_Corporate'] == 1:
                return 'Corporate'
            elif row['market segment type_Offline'] == 1:
                return 'Offline'
            elif row['market segment type_Online'] == 1:
                return 'Online'
            else:
                return 'Other'

        df['Market Segment Type'] = df.apply(map_market_segment, axis=1)

        view_limit = df[['lead time', 'average price', 'special requests','Market Segment Type', 'prediction']].head(10).to_dict(orient='records')

        return render_template('batch_result.html', view_limit=view_limit, output_file=output_file)

    return render_template('batch.html')

@app.route('/download')
def download_file():
    output_file = "predictions.csv"
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
