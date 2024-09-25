import os.path
import pandas as pd
import joblib


def feature_engineering(df:pd.DataFrame) -> pd.DataFrame:
    df['total_nights'] = df['number of week nights'] + df['number of weekend nights']
    df['average_price_per_night'] = (df['average price '] / df['total_nights']).round(2)
    df['total_guest'] = df['number of adults'] + df['number of children']
    df['average_price_per_night_per_guest'] = (df['average_price_per_night'] / df['total_guest']).round(2)
    return df


def outlier_treatment(df:pd.DataFrame, features_name: list)->pd.DataFrame:
    upper_lower = joblib.load(os.path.join(os.getcwd(), 'dict', 'upper_lower.pkl'))
    for feature in features_name:
        lower_bound = upper_lower['lower_bound'][feature]
        upper_bound = upper_lower['upper_bound'][feature]
        if len(df[feature][(df[feature] < lower_bound)]) != 0:
            df[feature][(df[feature] < lower_bound)] = lower_bound
        if len(df[feature][(df[feature] > upper_bound)]) != 0:
            df[feature][(df[feature] > upper_bound)] = upper_bound
    return df


def num_feature_selection(df: pd.DataFrame)->pd.DataFrame:
    selected_features = ['total_nights', 'average_price_per_night_per_guest',
                         'total_guest', 'lead time', 'special requests',
                         'P-C', 'P-not-C', 'repeated', 'car parking space']
    return df[selected_features]