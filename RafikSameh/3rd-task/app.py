from flask import Flask,request,jsonify,render_template
import joblib
import pickle
import numpy as np
import pandas as pd

app =  Flask(__name__)

model = joblib.load('Knn-fined-tuned-clf.pkl')
pipe = joblib.load('pipeline.pkl')
cols = ['type of meal', 'car parking space', 'room type', 'lead time',
       'market segment type', 'repeated', 'average price ', 'special requests',
       'total nights', 'month']
@app.route('/')
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = [
        request.form['type of meal'],
        request.form['car parking space'],
        request.form['room type'],
        request.form['lead time'],
        request.form['market segment type'],
        request.form['repeated'],
        request.form['average price'],
        request.form['special requests'],
        request.form['total nights'],
        request.form['month']
    ]
    
    input_features = pd.DataFrame([data],columns=cols)
    print(input_features)
    x_processed = pipe.transform(input_features)
    prediction = model.predict(x_processed)

    return render_template("index.html",prediction_text = "{}".format(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)
