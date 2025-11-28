# Heart Disease Prediction App

This is a Flask web application for predicting heart disease based on patient medical data.

## Features

- Web interface for inputting patient data
- Real-time prediction using trained machine learning model
- Support for multiple ML models (Logistic Regression, Random Forest, SVM)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Model

The app uses a trained model located in the `../Model` directory. The model is trained on the heart disease dataset and can predict the likelihood of heart disease based on 13 medical features.

## Input Features

The application accepts the following patient data:
- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Max Heart Rate (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope
- Number of Major Vessels (ca)
- Thalassemia (thal)
