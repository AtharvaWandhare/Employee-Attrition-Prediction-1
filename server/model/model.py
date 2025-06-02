import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# dtaa loading
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'HR-Employee-Attrition.csv')
df = pd.read_csv(data_path)

target = 'Attrition'
features = df.drop(target, axis=1)

categorical_features = features.select_dtypes(include=['object']).columns.tolist()
numerical_features = features.select_dtypes(exclude=['object']).columns.tolist()

le = LabelEncoder()
df[target] = le.fit_transform(df[target])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42, stratify=df[target])
model = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

def predict_attrition(pipeline: Pipeline, new_data: pd.DataFrame, return_probability: bool = False) -> pd.Series:
    if return_probability:
        probabilities = pipeline.predict_proba(new_data)
        attrition_probabilities = probabilities[:, 1]
        return pd.Series(attrition_probabilities)
    else:
        predictions = pipeline.predict(new_data)
        return pd.Series(predictions)