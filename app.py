from flask import Flask, render_template, request, flash, redirect
import os
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model

from generate_mocks import (
    generate_heuristic_diabetes_model,
    generate_heuristic_cancer_model,
    generate_heuristic_heart_model,
    generate_heuristic_kidney_model,
    generate_heuristic_liver_model,
)
from cnn_mocks import ensure_malaria_model, ensure_pneumonia_model


app = Flask(__name__)


def load_or_generate(model_path, generator=None):
    """
    Try to load a pickle model, and if it is missing, optionally
    generate a heuristic model first using the provided generator.
    """
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        if generator is not None:
            generator()
            with open(model_path, "rb") as f:
                return pickle.load(f)
        raise


def preprocess_form_values(form_dict):
    """
    Convert raw form values (including yes/no, normal/abnormal, etc.)
    into numeric features suitable for the heuristic models.
    """
    processed = []

    for key, raw_value in form_dict.items():
        value_str = str(raw_value).strip()
        lower = value_str.lower()

        # Kidney disease categorical fields
        if key in {"rbc", "pc"}:
            if lower == "normal":
                processed.append(0.0)
            elif lower == "abnormal":
                processed.append(1.0)
            else:
                raise ValueError(f"Invalid value for {key}: {value_str}")
        elif key in {"pcc", "ba"}:
            if lower in {"present", "yes", "y", "1"}:
                processed.append(1.0)
            elif lower in {"notpresent", "no", "n", "0"}:
                processed.append(0.0)
            else:
                raise ValueError(f"Invalid value for {key}: {value_str}")
        elif key in {"htn", "dm", "cad", "pe", "ane"}:
            if lower in {"yes", "y", "1"}:
                processed.append(1.0)
            elif lower in {"no", "n", "0"}:
                processed.append(0.0)
            else:
                raise ValueError(f"Invalid value for {key}: {value_str}")
        else:
            # All other fields are expected to be numeric
            processed.append(float(value_str))

    return processed


def interpret_risk(probability: float):
    """
    Map a raw probability (0-1) to a percentage, textual level,
    and a CSS class for styling.
    """
    prob = max(0.0, min(1.0, float(probability)))
    risk_percentage = round(prob * 100.0, 2)

    if risk_percentage <= 30:
        risk_level = "Low Risk"
        risk_class = "low"
    elif risk_percentage <= 70:
        risk_level = "Moderate Risk"
        risk_class = "moderate"
    else:
        risk_level = "High Risk"
        risk_class = "high"

    return risk_percentage, risk_level, risk_class


def predict(values, dic):
    """
    Select the appropriate model based on feature length,
    compute disease probability, and convert it into a
    risk percentage and level.
    """
    values = np.asarray(values)

    if len(values) == 8:
        model = load_or_generate(
            "models/diabetes.pkl", generate_heuristic_diabetes_model
        )
    elif len(values) == 26:
        model = load_or_generate(
            "models/breast_cancer.pkl", generate_heuristic_cancer_model
        )
    elif len(values) == 13:
        model = load_or_generate("models/heart.pkl", generate_heuristic_heart_model)
    elif len(values) == 18:
        model = load_or_generate("models/kidney.pkl", generate_heuristic_kidney_model)
    elif len(values) == 10:
        model = load_or_generate("models/liver.pkl", generate_heuristic_liver_model)
    else:
        raise ValueError(f"Unexpected number of features: {len(values)}")

    # Use predict_proba to obtain class-1 (disease) probability.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(values.reshape(1, -1))[0][1]
    else:
        # Fallback: approximate using binary prediction (0 or 1).
        proba = float(model.predict(values.reshape(1, -1))[0])

    return interpret_risk(proba)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = preprocess_form_values(to_predict_dict)
        risk_percentage, risk_level, risk_class = predict(
            to_predict_list, to_predict_dict
        )
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template(
        "predict.html",
        risk_percentage=risk_percentage,
        risk_level=risk_level,
        risk_class=risk_class,
    )

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' not in request.files or request.files['image'].filename == '':
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)

        img = Image.open(request.files['image'])
        img = img.resize((36, 36))
        img = np.asarray(img)
        img = img.reshape((1, 36, 36, 3))
        img = img.astype(np.float64)

        model_path = "models/malaria.keras"
        ensure_malaria_model(model_path)
        model = load_model(model_path)
        probs = model.predict(img)[0]
        # Assume binary softmax [p(healthy), p(disease)]
        disease_proba = float(probs[1]) if len(probs) > 1 else float(probs[0])
        risk_percentage, risk_level, risk_class = interpret_risk(disease_proba)

        return render_template(
            "malaria_predict.html",
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            risk_class=risk_class,
        )

    except:
        message = "There was an error processing the image."
        return render_template('malaria.html', message=message)


@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' not in request.files or request.files['image'].filename == '':
            message = "Please upload an Image"
            return render_template('pneumonia.html', message=message)

        img = Image.open(request.files['image']).convert('L')
        img = img.resize((36, 36))
        img = np.asarray(img)
        img = img.reshape((1, 36, 36, 1))
        img = img / 255.0

        model_path = "models/pneumonia.keras"
        ensure_pneumonia_model(model_path)
        model = load_model(model_path)
        probs = model.predict(img)[0]
        disease_proba = float(probs[1]) if len(probs) > 1 else float(probs[0])
        risk_percentage, risk_level, risk_class = interpret_risk(disease_proba)

        return render_template(
            "pneumonia_predict.html",
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            risk_class=risk_class,
        )

    except:
        message = "There was an error processing the image."
        return render_template('pneumonia.html', message=message)

if __name__ == '__main__':
	app.run(debug = True)