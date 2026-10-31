import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/credit-screening/crx.data"
)

COLUMNS = [f"A{i}" for i in range(1, 16)] + ["A16"]

CATEGORICAL = [
    "A1", "A4", "A5", "A6", "A7",
    "A9", "A10", "A12", "A13"
]

NUMERIC = ["A2", "A3", "A8", "A11", "A14", "A15"]


def train_model():
    print("Loading Credit Approval dataset...")

    df = pd.read_csv(
        DATA_URL,
        names=COLUMNS,
        na_values="?"
    )

    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df[df["A16"].isin(["+", "-"])].copy()

    X = df[CATEGORICAL + NUMERIC].copy()

    for col in CATEGORICAL:
        X[col] = X[col].astype("object")

    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # + means approved; - means not approved
    y = (df["A16"] == "+").astype(int)

    print("Dataset shape:", df.shape)
    print("Approved applications:", int(y.sum()))
    print("Not approved applications:", int((y == 0).sum()))

    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    # Complete machine learning pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Logistic Regression...")
    print("Optimizing hyperparameters with GridSearchCV...")

    parameters = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
        "classifier__solver": ["liblinear"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, probabilities), 4
        ),
        "best_parameters": grid_search.best_params_,
        "best_cv_f1": round(grid_search.best_score_, 4)
    }

    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/credit_model.pkl")

    with open("model/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL TRAINING COMPLETED")
    print("------------------------------")
    print("Best parameters:", grid_search.best_params_)
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])
    print("ROC-AUC:", metrics["roc_auc"])
    print("------------------------------")
    print("Model saved successfully!")
    print("File: model/credit_model.pkl")


if __name__ == "__main__":
    train_model()