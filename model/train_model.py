"""
Model training pipeline for Fake News Detection.
Uses Scikit-Learn TF-IDF Vectorizer and Calibrated PassiveAggressive / Logistic Regression Classifier.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_fake_news_model(dataset_path="data/news_dataset.csv", model_dir="model"):
    print("--- Starting Fake News Detection Model Training ---")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run data/create_dataset.py first.")

    df = pd.read_csv(dataset_path)
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

    X = df['combined_text']
    y = df['label']

    print(f"Total samples: {len(df)} | Real (1): {sum(y==1)} | Fake (0): {sum(y==0)}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer setup
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Classifier: LogisticRegression with TF-IDF Features
    model = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # Predictions & Metrics
    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Confusion Matrix: {cm}")

    # Cross-validation score on full dataset
    full_vec = vectorizer.transform(X)
    cv_scores = cross_val_score(model, full_vec, y, cv=5, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    print(f"5-Fold CV Accuracy: {cv_mean:.4f} (+/- {np.std(cv_scores):.4f})")

    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fake_news_model.joblib")
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    metrics_path = os.path.join(model_dir, "model_metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics_data = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "cv_accuracy": round(cv_mean, 4),
        "confusion_matrix": cm,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "total_training_samples": len(df),
        "model_type": "LogisticRegression (TF-IDF Unigrams + Bigrams)"
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"\nArtifacts successfully saved to '{model_dir}' directory.")
    return metrics_data

if __name__ == "__main__":
    train_fake_news_model()
