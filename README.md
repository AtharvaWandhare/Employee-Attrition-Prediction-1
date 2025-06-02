# Employee Attrition Prediction & Analysis

This project is a web application designed to predict employee attrition using machine learning and provide insightful visualizations on factors contributing to attrition. It is built with a Flask backend and a user-friendly web interface.

## Overview

Employee attrition is a critical concern for organizations. This application leverages a machine learning model trained on HR data to predict the likelihood of an employee leaving the company. Additionally, it offers an insights section where various data visualizations highlight key trends and factors related to employee attrition.

## Features

*   **Attrition Prediction:** Predicts whether an employee is likely to leave based on various input features.
*   **Data-driven Insights:** Visualizes relationships between employee attrition and factors like department, age, salary, overtime, job role, etc.
*   **Feature Importance:** Shows the most influential factors in predicting attrition.
*   **Interactive Web Interface:** Easy-to-use interface for prediction and exploring insights.

## Dataset

The project uses the "HR Employee Attrition" dataset.
*   **Dataset file:** [server/data/HR-Employee-Attrition.csv](server/data/HR-Employee-Attrition.csv)
*   **Source:** IBM HR Analytics Employee Attrition & Performance on Kaggle.

## Technologies Used

*   **Backend:**
    *   [Flask](https://flask.palletsprojects.com/): A micro web framework for Python.
*   **Machine Learning & Data Analysis:**
    *   [scikit-learn](https://scikit-learn.org/stable/): For machine learning model training and evaluation (Random Forest Classifier).
    *   [pandas](https://pandas.pydata.org/): For data manipulation and analysis.
    *   [NumPy](https://numpy.org/): For numerical operations.
*   **Data Visualization:**
    *   [Matplotlib](https://matplotlib.org/): For creating static, animated, and interactive visualizations.
    *   [Seaborn](https://seaborn.pydata.org/): For statistical data visualization.
*   **Frontend:**
    *   HTML
    *   CSS
    *   JavaScript (implied for dynamic interactions)

## Project Structure

```
Employee-Attrition-Prediction/
├── server/
│   ├── app.py                   # Main Flask 
|   |
│   ├── data/
│   │   └── HR-Employee-Attrition.csv
|   |
│   ├── model/
│   │   └── model.py             # Machine learning model training
|   |
│   ├── static/                  # Static files (CSS, JavaScript, images)
│   │   ├── css/
│   │   │   ├── index.css
│   │   │   ├── insights.css
│   │   │   ├── nav.css
│   │   │   ├── predict.css
│   │   │   └── util.css
│   │   └── images/
│   ├── templates/               # HTML templates
│   │   ├── index.html           # Home page
│   │   └── pages/
│   │       ├── predict.html     # Prediction page
│   │       ├── insights.html    # Insights page
│   │       └── about.html       # About page (inferred)
│   └── __pycache__/
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/WandhareAtharva/Employee-Attrition-Prediction.git
    cd Employee-Attrition-Prediction
    ```

## Running the Application

1.  **Navigate to the server directory:**
    ```bash
    cd server
    ```

2.  **Run the Flask application:**
    You might be using either [`app.py`](server/app.py) or [`app_fixed.py`](server/app_fixed.py).
    ```bash
    python app.py
    ```

3.  **Open your web browser** and go to: `http://127.0.0.1:5000/`

## Pages

*   **Home (`/`):** [server/templates/index.html](server/templates/index.html)
    *   Introduction to the application with links to predict attrition and view insights.
*   **Predict (`/predict`):** [server/templates/pages/predict.html](server/templates/pages/predict.html) or [server/templates/pages/predict_fixed.html](server/templates/pages/predict_fixed.html)
    *   A form to input employee details.
    *   Submitting the form provides a prediction on whether the employee is likely to attrite.
*   **Insights (`/insights`):** [server/templates/pages/insights.html](server/templates/pages/insights.html)
    *   Displays various visualizations and analyses related to employee attrition, such as:
        *   Attrition by department, age, salary, overtime.
        *   Feature importance plot.
*   **About (`/about`):** (Inferred from navigation links)
    *   Likely contains information about the project or creators.