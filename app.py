from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model/credit_model.pkl"

CATEGORICAL = [
    "A1", "A4", "A5", "A6", "A7",
    "A9", "A10", "A12", "A13"
]

NUMERIC = ["A2", "A3", "A8", "A11", "A14", "A15"]

FEATURES = CATEGORICAL + NUMERIC

CATEGORIES = {
    "A1": ["a", "b"],
    "A4": ["u", "y", "l", "t"],
    "A5": ["g", "p", "gg"],
    "A6": [
        "c", "d", "cc", "i", "j", "k", "m",
        "r", "q", "w", "x", "e", "aa", "ff"
    ],
    "A7": [
        "v", "h", "bb", "j", "n", "z",
        "dd", "ff", "o"
    ],
    "A9": ["t", "f"],
    "A10": ["t", "f"],
    "A12": ["t", "f"],
    "A13": ["g", "p", "s"]
}


@app.route("/")
def home():
    return render_template("index.html", categories=CATEGORIES)


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Model missing. Run python train_model.py first."
        }), 500

    try:
        data = request.get_json(force=True)
        row = {}

        for col in CATEGORICAL:
            value = str(data.get(col, "")).strip().lower()

            if value not in CATEGORIES[col]:
                return jsonify({
                    "error": f"Invalid value for {col}."
                }), 400

            row[col] = value

        for col in NUMERIC:
            try:
                value = float(data.get(col, ""))
            except (TypeError, ValueError):
                return jsonify({
                    "error": f"Enter a number for {col}."
                }), 400

            if value < 0:
                return jsonify({
                    "error": f"{col} cannot be negative."
                }), 400

            row[col] = value

        X = pd.DataFrame([row], columns=FEATURES)

        model = joblib.load(MODEL_PATH)

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])

        return jsonify({
            "result": (
                "Likely Approved"
                if prediction == 1
                else "Likely Not Approved"
            ),
            "approval_probability": round(probability * 100, 2),
            "note": (
                "Educational prediction only; "
                "not a real credit decision."
            )
        })

    except Exception as error:
        app.logger.exception("Prediction error")
        return jsonify({
            "error": "Unable to process this prediction."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)