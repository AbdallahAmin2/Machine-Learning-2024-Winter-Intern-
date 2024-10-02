from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Class to convert the LeadTime feature into category

class convert_lead_time(BaseEstimator,TransformerMixin):
    def __init__(self,columns= None):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = pd.cut(X_copy[col],
                                 bins=[0, 30, 60, X_copy[col].max()],
                                 labels=[0, 1, 2],
                                 include_lowest=True)
        return X_copy[self.columns]


app = Flask(__name__)

# Load your machine learning model
model = joblib.load('best_rf_model.pkl')
pipeline = joblib.load('pipeline.pkl')

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the form data
       
        
        d = {'number of adults':float(request.form['number_of_adults']),
             'number of children':float(request.form['number_of_children']),
             'number of weekend nights':float(request.form['number_of_weekend_nights']),
            'number of week nights':float(request.form['number_of_week_nights']),
            'type of meal':request.form['type_of_meal'],
            'car parking space':request.form['car_parking_space'],
            'room type':request.form['room_type'],
            'lead time':float(request.form['lead_time']),
            'market segment type':request.form['market_segment_type'],
            'repeated':float(request.form['repeated']),
            'P-C':float(request.form['P_C']),
            'P-not-C':float(request.form['P_not_C']),
            'average price ':float(request.form['average_price']),
            'special requests':float(request.form['special_requests'])
                             }

        data = pd.DataFrame(d,index=[0])
        ready_data = pipeline.transform(data)
        # Make a prediction
        prediction = model.predict(ready_data)[0]
        

        
            
    return render_template('index2.html', prediction= 'Canceled' if prediction ==0 else 'Not Canceled')

# @app.route('/predict', methods=['GET'])
# def predict():
#     data = request.json
#     print("------------------------------------------")
#     print(data)
#     print("------------------------------------------")
    
#     # Convert the received data into a DataFrame
#     input_data = pd.DataFrame([data])
    
#     # applying the pipeline
#     ready_data = pipeline.transform(input_data)

#     # Predict using the ML model
#     prediction = model.predict(ready_data)
    
#     return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
