import pandas as pd
import joblib

def cancel_predict(input_data):
    # Load the pre-trained model
    model = joblib.load('random_forest_model (2).pkl')  # Adjust the path as necessary
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    input_df.columns = input_df.columns.str.strip()  # Strip whitespace
    input_df_encoded = pd.get_dummies(input_df)

    # Align with model's training features
    model_columns = [
        'number of adults', 'number of children', 'number of weekend nights',
        'number of week nights', 'car parking space', 'lead time', 'repeated',
        'P-C', 'P-not-C', 'average price ', 'special requests',
        'type of meal_Meal Plan 1', 'type of meal_Meal Plan 2',
        'type of meal_Meal Plan 3', 'type of meal_Not Selected',
        'room type_Room_Type 1', 'room type_Room_Type 2',
        'room type_Room_Type 3', 'room type_Room_Type 4',
        'room type_Room_Type 5', 'room type_Room_Type 6',
        'room type_Room_Type 7', 'market segment type_Aviation',
        'market segment type_Complementary', 'market segment type_Corporate',
        'market segment type_Offline', 'market segment type_Online'
    ]

    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df_encoded)
    
    return prediction
