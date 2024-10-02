from flask import Flask, render_template, url_for, redirect, request
from datetime import datetime
from model import preprocess_features, model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        form_data = get_form_data(request.form)
        
        if any(not value for value in form_data.values()):
            return redirect(url_for('predict', not_valid=True, **form_data))
        
        features = [form_data[key] for key in ['adults', 'children', 'weekend', 'week', 'meal_type', 
                                               'parking', 'room_type', 'lead_time', 
                                               'market', 'repeated', 
                                               'price', 'requests', 'month', 'day']]
        features = preprocess_features(features)
        print(features)
        
        prediction = model.predict(features)
        print(prediction)
    
        return redirect(url_for('predict', prediction=prediction[0], **form_data))
    
    not_valid = request.args.get('not_valid')
    prediction = request.args.get('prediction')
    if not not_valid and not prediction:
        return redirect(url_for('home'))
        
    form_data = get_form_data(request.args)

    return render_template('home.html', not_valid=not_valid, prediction=prediction, **form_data)



def get_form_data(source):
    reservation_date_str = source.get('reservation_date')
    
    return {
        'adults': source.get('adults'),
        'children': source.get('children'), 
        'weekend': source.get('weekend'),
        'week': source.get('week'),
        'meal_type': source.get('meal_type'),
        'parking': source.get('parking'),
        'room_type': source.get('room_type'),
        'lead_time': source.get('lead_time'),
        'market': source.get('market'),
        'repeated': source.get('repeated'),
        #'pc': source.get('pc'),
        #'pnotc': source.get('pnotc'),
        'reservation_date' : source.get('reservation_date'),
        'price': source.get('price'),
        'requests': source.get('requests'),
        'month': datetime.strptime(reservation_date_str, "%Y-%m-%d").month if reservation_date_str else None,
        'day': datetime.strptime(reservation_date_str, "%Y-%m-%d").day if reservation_date_str else None
    }
    
    

if __name__ == "__main__":
    app.run(debug=True)