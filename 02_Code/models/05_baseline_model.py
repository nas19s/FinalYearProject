"""
05_baseline_model.py

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score)
from sklearn.dummy import DummyClassifier

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "04_Results", "baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "final_feature_dataset.parquet")
TARGET_COL = "Label_T20"

FEATURE_COLS = [
    "Gunning_Fog", "Flesch_Ease", "Sentiment",
    "Diff_Word_Ratio", "Word_Count",
    "RSI", "MACD", "Volume_Change",
]


def evaluate_model(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(classification_report(y_true, y_pred,
                                 target_names=["DOWN", "UP"], zero_division=0))
    return {"model": name, "accuracy": acc, "f1_macro": f1, "auc": auc}


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def main():
    print("=" * 70)
    print("05_baseline_model.py  —  Baseline Classifiers")
    print("=" * 70)

    df = pd.read_parquet(INPUT_FILE)

    # Binary only — drop FLAT
    df = df[df[TARGET_COL] != 0].copy()
    df["binary_label"] = df[TARGET_COL].map({-1: 0, 1: 1})
    df = df.dropna(subset=["binary_label"] + FEATURE_COLS)

    print(f"Samples after dropping FLAT: {len(df):,}")
    print(f"Class balance:\n{df['binary_label'].value_counts()}")

    # Time-based split — match 08_train_finbert.py exactly
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    train = df[df["filing_date"] <  "2021-01-01"]
    test  = df[df["filing_date"] >= "2023-01-01"]

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["binary_label"]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test["binary_label"]

    print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")

    results = []

    # ── Baseline 1: Majority class (always predicts UP) ───────────────────
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    results.append(evaluate_model(
        "Majority Class (always UP)", y_test, y_pred_dummy
    ))
    plot_confusion_matrix(
        y_test, y_pred_dummy,
        "Majority Class Baseline",
        os.path.join(RESULTS_DIR, "baseline_majority_confusion.png")
    )

    # ── Baseline 2: Logistic Regression on engineered features ────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

    results.append(evaluate_model(
        "Logistic Regression (engineered features)", y_test, y_pred_lr, y_prob_lr
    ))
    plot_confusion_matrix(
        y_test, y_pred_lr,
        "Logistic Regression Baseline",
        os.path.join(RESULTS_DIR, "baseline_lr_confusion.png")
    )

    # ── Feature importance ─────────────────────────────────────────────────
    importance = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Coefficient": lr.coef_[0],
    }).sort_values("Coefficient", key=abs, ascending=False)
    print("\nLogistic Regression feature importance:")
    print(importance.to_string(index=False))
    importance.to_csv(
        os.path.join(RESULTS_DIR, "baseline_feature_importance.csv"), index=False
    )

    # ── Save summary ───────────────────────────────────────────────────────
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(RESULTS_DIR, "baseline_summary.csv"), index=False)
    print(f"\nSummary saved -> {RESULTS_DIR}")
    print("=" * 70)
    print("DONE — next step: 07_finbert_prep.py")
    print("=" * 70)


if __name__ == "__main__":
    main()