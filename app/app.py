from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"


def load_artifacts():
    mdl = None
    cols = None
    if MODEL_PATH.exists() and COLUMNS_PATH.exists():
        mdl = pickle.load(open(MODEL_PATH, "rb"))
        cols = pickle.load(open(COLUMNS_PATH, "rb"))
    return mdl, cols


model, model_columns = load_artifacts()


@app.route('/')
def home():
    if model is None or model_columns is None:
        # Do not use a template; just print and return plain text
        print("model not found")
        return "model not found", 503
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_columns is None:
        return redirect(url_for('home'))
    def to_float(name):
        return float(request.form.get(name))

    def to_int(name):
        return int(float(request.form.get(name)))

    fields = [
        'age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
    ]

    row = {
        'age': to_float('age'),
        'sex': to_int('sex'),
        'cp': to_int('cp'),
        'trestbps': to_float('trestbps'),
        'chol': to_float('chol'),
        'fbs': to_int('fbs'),
        'restecg': to_int('restecg'),
        'thalach': to_float('thalach'),
        'exang': to_int('exang'),
        'oldpeak': to_float('oldpeak'),
        'slope': to_int('slope'),
        'ca': to_int('ca'),
        'thal': to_int('thal'),
    }

    data = pd.DataFrame([row])
    data = data.reindex(columns=model_columns, fill_value=0)
    pred = model.predict(data)[0]
    prediction = 'disease' if int(pred) == 1 else 'no_disease'
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
