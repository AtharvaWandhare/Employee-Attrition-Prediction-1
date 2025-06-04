import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# dtaa loading
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'HR-Employee-Attrition.csv')
df = pd.read_csv(data_path)

target = 'Attrition'
features = df.drop(target, axis=1)

# Drop features that are not valuable
features = features.drop(['Over18', 'EmployeeNumber', 'EmployeeCount', 'StandardHours'], axis=1)

categorical_features = features.select_dtypes(include=['object']).columns.tolist()
numerical_features = features.select_dtypes(exclude=['object']).columns.tolist()

le = LabelEncoder()
df[target] = le.fit_transform(df[target])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42, stratify=df[target])
model = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X_train, y_train)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

print(f'Best parameters: {random_search.best_params_}')
print(f'Best cross-validation accuracy: {random_search.best_score_}')

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)

# Use the best estimator for predictions
y_pred = random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy (after tuning): {accuracy}')
# print(f'Classification Report (after tuning):\n{report}')

# Update pipeline to best estimator
pipeline = random_search.best_estimator_

def predict_attrition(pipeline: Pipeline, new_data: pd.DataFrame, return_probability: bool = False) -> pd.Series:
    if return_probability:
        probabilities = pipeline.predict_proba(new_data)
        attrition_probabilities = probabilities[:, 1]
        return pd.Series(attrition_probabilities)
    else:
        predictions = pipeline.predict(new_data)
        return pd.Series(predictions)