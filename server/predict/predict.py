import pandas as pd
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import predict_attrition, pipeline, features, categorical_features, numerical_features

def get_user_input_for_prediction(features_list, categorical_features, numerical_features):
    user_data = {}
    print("Please enter the values for each feature:")

    for feature in features_list:
        user_input = input(f"Enter value for '{feature}': ")

        if feature in numerical_features:
            try:
                user_data[feature] = [float(user_input)]
            except ValueError:
                print(f"Warning: Could not convert '{user_input}' to a number for feature '{feature}'. Please enter a valid number.")
                user_data[feature] = [None]
        else:
            user_data[feature] = [user_input]

    return pd.DataFrame(user_data)

features_list = features.columns.tolist()
new_employee_data = get_user_input_for_prediction(features_list, categorical_features, numerical_features)
new_employee_data = new_employee_data[features_list]

try:
    prediction = predict_attrition(pipeline, new_employee_data)
    print("\nPrediction:")
    predicted_attrition = "Yes" if prediction.iloc[0] == 1 else "No"
    print(f"The predicted attrition for this employee is: {predicted_attrition}")
except Exception as e:
    print(f"\nError during prediction: {e}")
    print("Please ensure all features were entered correctly and match the data used for training.")