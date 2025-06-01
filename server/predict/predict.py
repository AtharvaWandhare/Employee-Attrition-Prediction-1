import pandas as pd
from sklearn.pipeline import Pipeline
import sys
import os

# Add the parent directory to sys.path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the predict_attrition function and other variables from model.py
from model.model import predict_attrition, pipeline, features, categorical_features, numerical_features

def get_user_input_for_prediction(features_list, categorical_features, numerical_features):
    """
    Prompts the user to input values for each feature.

    Args:
        features_list: A list of all feature names.
        categorical_features: A list of categorical feature names.
        numerical_features: A list of numerical feature names.

    Returns:
        A pandas DataFrame containing the user's input.
    """
    user_data = {}
    print("Please enter the values for each feature:")

    for feature in features_list:
        user_input = input(f"Enter value for '{feature}': ")

        # Attempt to convert numerical features to float, handle errors
        if feature in numerical_features:
            try:
                user_data[feature] = [float(user_input)] # Input needs to be in a list for DataFrame
            except ValueError:
                print(f"Warning: Could not convert '{user_input}' to a number for feature '{feature}'. Please enter a valid number.")
                user_data[feature] = [None] # Or handle as an error or re-prompt
        else: # Assume categorical
            user_data[feature] = [user_input] # Input needs to be in a list for DataFrame

    return pd.DataFrame(user_data)

# The variables features, categorical_features, numerical_features, and pipeline are now imported from model.py

# Get the list of feature names
features_list = features.columns.tolist()

# Get user input
new_employee_data = get_user_input_for_prediction(features_list, categorical_features, numerical_features)

# Ensure the input DataFrame has the same columns and order as the training features
# This step is crucial for the pipeline to work correctly
new_employee_data = new_employee_data[features_list]


# Make predictions
try:
    prediction = predict_attrition(pipeline, new_employee_data) # Reuse the predict_attrition function
    print("\nPrediction:")
    # Assuming the target was encoded as 0/1 and 1 corresponds to 'Yes' Attrition
    predicted_attrition = "Yes" if prediction.iloc[0] == 1 else "No"
    print(f"The predicted attrition for this employee is: {predicted_attrition}")
except Exception as e:
    print(f"\nError during prediction: {e}")
    print("Please ensure all features were entered correctly and match the data used for training.")