"""
train_model.py
--------------
Trains a Random Forest classifier on URL features to detect phishing pages.
Evaluates model performance and saves the trained model to disk.

Pipeline:
  generate_dataset → extract features → train/test split →
  Random Forest → evaluate → save model

Author : Mohammed Furqan Sajid
Project: Phishing URL Detection using Machine Learning
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing     import StandardScaler

from feature_extractor  import FEATURE_NAMES
from generate_dataset   import build_dataset


# ── configuration ─────────────────────────────────────────────────────────────

MODEL_PATH   = "phishing_model.pkl"
SCALER_PATH  = "scaler.pkl"
RANDOM_STATE = 42
TEST_SIZE    = 0.2       # 80% train / 20% test
N_SAMPLES    = 1000      # total samples (500 legit + 500 phish)


# ── helper: pretty print section ─────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ── training pipeline ─────────────────────────────────────────────────────────

def train():
    # 1. Build / load dataset
    section("Step 1 — Building dataset")
    df = build_dataset(N_SAMPLES // 2, N_SAMPLES // 2)
    print(f"Total samples : {len(df)}")
    print(f"Label balance : {df['label'].value_counts().to_dict()}  (0=legit, 1=phish)")

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # 2. Train / test split
    section("Step 2 — Splitting data (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set  : {X_train.shape[0]} samples")
    print(f"Test set      : {X_test.shape[0]} samples")

    # 3. Feature scaling
    section("Step 3 — Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("StandardScaler fitted on training data only (no data leakage)")

    # 4. Train Random Forest
    section("Step 4 — Training Random Forest classifier")
    model = RandomForestClassifier(
        n_estimators=100,       # 100 decision trees
        max_depth=None,         # grow trees fully
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1               # use all CPU cores
    )
    model.fit(X_train_scaled, y_train)
    print("Random Forest trained with 100 estimators")

    # 5. Cross-validation
    section("Step 5 — 5-fold Cross-Validation")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Evaluation on test set
    section("Step 6 — Test Set Evaluation")
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives  (legit   → legit  ) : {cm[0][0]}")
    print(f"  False Positives (legit   → phishing) : {cm[0][1]}")
    print(f"  False Negatives (phishing→ legit  ) : {cm[1][0]}")
    print(f"  True Positives  (phishing→ phishing) : {cm[1][1]}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # 7. Feature importance
    section("Step 7 — Feature Importance (top 10)")
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1], reverse=True
    )
    for rank, (name, imp) in enumerate(feat_imp[:10], 1):
        bar = "█" * int(imp * 100)
        print(f"  {rank:2}. {name:<30} {imp:.4f}  {bar}")

    # 8. Save model and scaler
    section("Step 8 — Saving model")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")

    return model, scaler, acc


if __name__ == "__main__":
    train()
