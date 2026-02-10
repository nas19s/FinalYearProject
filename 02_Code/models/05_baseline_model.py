import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
INPUT_FILE = os.path.join(DATA_DIR, "final_feature_dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "baseline")

# Create results folder
os.makedirs(RESULTS_DIR, exist_ok=True)

# TARGET CHOICE: 
# 'Label_Month' (Medium Term) had the most signals (approx 600).
# We use this as our primary target.
TARGET_COL = 'Label_Month' 

def run_baseline():
    print(f"Starting Baseline Model (Target: {TARGET_COL})...")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print("Error: Dataset not found.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"   Original Rows: {len(df)}")

    # 2. IMPLEMENT OPTION A: Filter out Neutrals (-1)
    # We only want rows where Label is 0 (Down) or 1 (Up)
    df_clean = df[df[TARGET_COL] != -1].copy()
    
    print(f"   Rows after removing Neutrals: {len(df_clean)}")
    print(f"   Class Balance:\n{df_clean[TARGET_COL].value_counts()}")

    # 3. Define Features (X) and Target (y)
    # We use the engineered features + technical indicators
    feature_cols = [
        'Gunning_Fog', 'Flesch_Ease', 'Sentiment', 
        'Diff_Word_Ratio', 'Word_Count',
        'RSI', 'MACD', 'Volume_Change'
    ]
    
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    # Handle any remaining NaNs (just in case)
    X = X.fillna(0)

    # 4. Split Data (80% Train, 20% Test)
    # stratify=y ensures we have equal mix of Up/Down in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # 5. Train Logistic Regression (The "Standard" Baseline)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Baseline Accuracy: {acc:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 7. Save Confusion Matrix (Evidence for Report)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f"Baseline Confusion Matrix ({TARGET_COL})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    save_path = os.path.join(RESULTS_DIR, "baseline_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"   Confusion Matrix saved to: {save_path}")

    # 8. Feature Importance (What mattered most?)
    # Logistic Regression coefficients show us what drove the decision
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', key=abs, ascending=False)
    
    print("\n--- Feature Importance (Top Drivers) ---")
    print(importance)

if __name__ == "__main__":
    run_baseline()