"""
Train and evaluate three creditworthiness classifiers.
Dataset note: credit_data.csv is synthetic teaching data generated for this project.
It is NOT real lending data and should not be used for actual credit decisions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.inspection import permutation_importance
import joblib
from feature_engineering import (
    INPUT_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    MODEL_FEATURES, add_engineered_features,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "creditworthy"

def main():
    data = pd.read_csv(DATA_PATH)
    # Add a derived financial ratio from the original input columns.
    X = add_engineered_features(data[INPUT_FEATURES])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=4,
            random_state=42, class_weight="balanced"
        )
    }

    results = {}
    fitted = {}
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test, predictions, target_names=["Not creditworthy", "Creditworthy"],
                zero_division=0, output_dict=True
            )
        }
        fitted[name] = pipeline
        joblib.dump(pipeline, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))

    # Save test split for consistent evaluation visualizations
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv(MODEL_DIR / "test_data.csv", index=False)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Training completed.")
    print("Dataset: synthetic teaching data only.")
    print("Feature engineering: added debt_to_annual_income_ratio = total_debt / annual_income.")
    for name, metrics in results.items():
        print(f"\n{name}")
        print("  Accuracy :", f"{metrics['accuracy']:.3f}")
        print("  Precision:", f"{metrics['precision']:.3f}")
        print("  Recall   :", f"{metrics['recall']:.3f}")
        print("  F1-score :", f"{metrics['f1_score']:.3f}")
        print("  ROC-AUC  :", f"{metrics['roc_auc']:.3f}")
    print(f"\nModel files and metrics saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
