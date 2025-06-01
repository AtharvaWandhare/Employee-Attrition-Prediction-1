import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Changed to RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report # Changed metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Added LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the data - use absolute path for better reliability
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'HR-Employee-Attrition.csv')
df = pd.read_csv(data_path)

# 2. Identify target variable and features
target = 'Attrition'
features = df.drop(target, axis=1)

# Identify categorical and numerical features
categorical_features = features.select_dtypes(include=['object']).columns.tolist()
numerical_features = features.select_dtypes(exclude=['object']).columns.tolist()

# 3. Handle categorical data
# For the target variable 'Attrition', we'll use LabelEncoder to convert it to numerical (0 or 1)
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42, stratify=df[target]) # Added stratify

# 5. Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42) # Changed to RandomForestClassifier

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)]) # Changed 'regressor' to 'classifier'

pipeline.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = pipeline.predict(X_test)

# Evaluate using classification metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

import pandas as pd
from sklearn.pipeline import Pipeline

def predict_attrition(pipeline: Pipeline, new_data: pd.DataFrame, return_probability: bool = False) -> pd.Series:
    """
    Uses a trained pipeline to predict attrition on new data.

    Args:
        pipeline: The trained scikit-learn pipeline containing preprocessing and the model.
        new_data: A pandas DataFrame with the same feature columns as the training data,
                  excluding the target variable.
        return_probability: If True, returns the probability of attrition (class 1) instead of binary prediction.

    Returns:
        A pandas Series containing either:
        - The predicted attrition labels (0 or 1) if return_probability is False
        - The probability of attrition (0.0-1.0) if return_probability is True
    """
    if return_probability:
        # The pipeline's predict_proba method returns probabilities for each class
        # We want the probability for class 1 (attrition = Yes), which is at index 1
        probabilities = pipeline.predict_proba(new_data)
        attrition_probabilities = probabilities[:, 1]  # Get probabilities for class 1 (attrition)
        return pd.Series(attrition_probabilities)
    else:
        # Return binary predictions (0 or 1)
        predictions = pipeline.predict(new_data)
        return pd.Series(predictions)

# Example usage (assuming you have already run the previous code to train the pipeline):
# Let's create some sample new data (replace this with your actual new data)
# Ensure the columns in new_data match the feature columns used for training
sample_new_data = features.sample(5, random_state=42) # Using a sample of the original features for demonstration

# Make predictions on the new data - binary predictions (0 or 1)
new_predictions = predict_attrition(pipeline, sample_new_data)

# Make probability predictions (0.0-1.0)
probability_predictions = predict_attrition(pipeline, sample_new_data, return_probability=True)

print("Binary predictions on new data:")
print(new_predictions)
print("\nProbability predictions on new data (chance of attrition):")
print(probability_predictions.map(lambda x: f"{x:.2%}"))