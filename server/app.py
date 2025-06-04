from flask import Flask, url_for, render_template, request, send_from_directory, jsonify
import pandas as pd
import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ya code ni current directory ni import hota fakt, konte pan files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import predict_attrition, pipeline, features, categorical_features, numerical_features

app = Flask(__name__)
app.secret_key = "employee_attrition_prediction"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    if request.method == "POST":
        try:
            user_data = {}
            # print("Received POST request with form data:", request.form)
            for field in request.form:
                val = request.form.get(field)
                if field in numerical_features:
                    try:
                        user_data[field] = [float(val)]
                    except ValueError:
                        # flash(f"Invalid value for {field}. Please enter a number.")
                        return render_template("pages/predict.html")
                else:
                    user_data[field] = [val.strip()]
            
            for feature in features.columns:
                if feature not in user_data:
                    user_data[feature] = [features[feature].median() if feature in numerical_features else features[feature].mode()[0]]

            input_df = pd.DataFrame(user_data)[features.columns]
            pred = predict_attrition(pipeline, input_df)
            prediction = "Yes" if pred.iloc[0] == 1 else "No"
            
            prob = predict_attrition(pipeline, input_df, return_probability=True)
            probability = round(float(prob.iloc[0]) * 100, 2)

        except Exception as e:
            print(f"Error making prediction: {e}")
            # flash(f"An error occurred: {e}")
            return render_template("pages/predict.html")

    return render_template("pages/predict.html", result=prediction is not None, prediction=prediction, probability=probability)


@app.route("/about")
def about():
    return render_template("pages/about.html")

@app.route("/insights")
def insights():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/HR-Employee-Attrition.csv")
    
    df = pd.read_csv(data_path)
    total_employees = len(df)
    left_count = df[df['Attrition'] == 'Yes'].shape[0]
    stayed_count = total_employees - left_count
    attrition_rate = (left_count / total_employees) * 100
    
    feature_importance_plot = generate_plot_base64(generate_feature_importance_plot, df)

    all_available_features = features.columns.tolist()
    selected_feature = request.args.get('feature')
    selected_feature_plot_data = None

    if selected_feature and selected_feature in all_available_features:
        selected_feature_plot_data = generate_plot_base64(
            lambda df_lambda: generate_dynamic_feature_plot(df_lambda, selected_feature, numerical_features, categorical_features), 
            df
        )
    
    return render_template("pages/insights.html", 
                           total_employees=total_employees,
                           left_count=left_count,
                           stayed_count=stayed_count,
                           attrition_rate=round(attrition_rate, 2),
                           feature_importance=feature_importance_plot,
                           all_features=all_available_features,
                           selected_feature=selected_feature,
                           selected_feature_plot_data=selected_feature_plot_data)

def generate_dynamic_feature_plot(df, feature_name, numerical_features_list, categorical_features_list):
    """Generates a plot for a given feature against attrition."""
    plt.figure(figsize=(10, 6))
    if feature_name in numerical_features_list:
        # For numerical features -> histogram without KDE
        sns.histplot(data=df, x=feature_name, hue='Attrition', kde=False, multiple="stack", palette="Set2")
        plt.title(f'Attrition Distribution by {feature_name}', fontsize=14)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Count', fontsize=12)
    elif feature_name in categorical_features_list:
        # categorical features -> count plot
        graph = sns.countplot(data=df, x=feature_name, hue='Attrition', palette="Set2")
        plt.title(f'Attrition by {feature_name}', fontsize=14)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        if df[feature_name].nunique() > 5 and df[feature_name].dtype == 'object':
            plt.xticks(rotation=45, ha='right')
        
        for p in graph.patches:
            graph.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points')
    else:
        plt.text(0.5, 0.5, f"Cannot generate plot for feature: {feature_name}. Type unknown.",
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, fontsize=12, color='red')

def generate_plot_base64(plot_function, df):
    """Generate a base64-encoded plot using the specified function."""
    plt.figure(figsize=(10, 6))
    plot_function(df)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def generate_feature_importance_plot(df):
    df = df.copy()

    for col in df.select_dtypes(include='object').columns:
        if col != 'Attrition':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    if 'Attrition' not in df.columns:
        print("Error: 'Attrition' column not found.")
        return

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df = df.dropna(subset=['Attrition'])

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances * 100
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=True).tail(10)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance (%)')
    plt.title('Top 10 Features Affecting Attrition')
    plt.tight_layout()
    
@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

@app.route("/dataset")
def dataset():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/HR-Employee-Attrition.csv")
    df = pd.read_csv(data_path)
    
    table_html = df.head(1470).to_html(classes="table table-striped table-bordered", index=False)
    return render_template("pages/dataset.html", table_html=table_html)

if __name__ == "__main__":
    app.run(debug=True)