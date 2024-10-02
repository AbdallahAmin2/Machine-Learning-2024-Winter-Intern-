from flask import Flask, render_template, request
import RF

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/sub", methods=["POST"])
def submit():
    if request.method == "POST":
        # Collect input data as a dictionary
        input_data = {
            'number of adults': request.form.get('number_of_adults', type=int),
            'number of children': request.form.get('number_of_children', type=int),
            'number of weekend nights': request.form.get('number_of_weekend_nights', type=int),
            'number of week nights': request.form.get('number_of_week_nights', type=int),
            'car parking space': request.form.get('car_parking_space', type=int),
            'lead time': request.form.get('lead_time', type=int),
            'repeated': request.form.get('repeated', type=int),
            'P-C': request.form.get('P_C', type=int),
            'P-not-C': request.form.get('P_not_C', type=int),
            'average price ': request.form.get('average_price', type=float),  # Note the space
            'special requests': request.form.get('special_requests', type=int),
            # Room types
            **{f'room type_Room_Type {i}': request.form.get(f'room_type_{i}', type=int) for i in range(1, 8)},
            # Market segments
            **{f'market segment type_{segment}': request.form.get(f'market_segment_{segment}', type=int) for segment in ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online']},
            # Meal plans
            **{f'type of meal_Meal Plan {i}': request.form.get(f'meal_plan_{i}', type=int) for i in range(1, 4)},
            'type of meal_Not Selected': request.form.get('meal_not_selected', type=int),
        }
        
        y_pred_forest = RF.cancel_predict(input_data)
        
    return render_template("sub.html", prediction=y_pred_forest)

if __name__ == "__main__":
    app.run(debug=True)
