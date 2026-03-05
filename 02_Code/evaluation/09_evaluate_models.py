import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Ensure non-interactive backend for server/script use
matplotlib.use("Agg")

# Project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "metrics")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "04_Results", "baseline")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_predictions(csv_path, model_name):
    """Calculate accuracy, F1, and AUC from prediction CSVs."""
    df = pd.read_csv(csv_path)
    y_true = df["true_label"].astype(int)
    y_pred = df["pred_label"].astype(int)
    y_prob = df["prob_up"].astype(float)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")

    down_f1 = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    up_f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "F1_Macro": round(f1, 4),
        "AUC": round(auc, 4),
        "F1_DOWN": round(down_f1, 4),
        "F1_UP": round(up_f1, 4),
        "N_test": len(df),
    }

def main():
    rows = []

    # Process FinBERT results
    horizons = ["T5", "T10", "T20"]
    for h in horizons:
        csv_path = os.path.join(RESULTS_DIR, f"finbert_test_predictions_{h}.csv")
        if os.path.exists(csv_path):
            rows.append(evaluate_predictions(csv_path, f"FinBERT ({h})"))
            print(f"Processed FinBERT {h}")
        else:
            print(f"File missing: {csv_path}")

    # Load previously saved baseline metrics
    baseline_path = os.path.join(BASELINE_DIR, "baseline_summary.csv")
    if os.path.exists(baseline_path):
        bl_df = pd.read_csv(baseline_path)
        for _, r in bl_df.iterrows():
            rows.append({
                "Model": r["model"],
                "Accuracy": round(r.get("accuracy", float("nan")), 4),
                "F1_Macro": round(r.get("f1_macro", float("nan")), 4),
                "AUC": round(r.get("auc", float("nan")), 4),
                "F1_DOWN": float("nan"),
                "F1_UP": float("nan"),
                "N_test": "—",
            })
        print("Baseline metrics loaded.")

    # Create master table
    results = pd.DataFrame(rows)
    results = results.sort_values("F1_Macro", ascending=False)

    print("\nMaster Results:")
    print(results.to_string(index=False))

    csv_out = os.path.join(RESULTS_DIR, "master_results_table.csv")
    results.to_csv(csv_out, index=False)
    print(f"Saved table to: {csv_out}")

    # Visualization
    finbert_data = results[results["Model"].str.startswith("FinBERT")]
    if finbert_data.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("FinBERT Performance by Horizon", fontsize=12, fontweight="bold")
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # F1 Plot
    ax1.bar(finbert_data["Model"], finbert_data["F1_Macro"], color=colors)
    ax1.axhline(0.50, color="red", linestyle="--", label="Random (0.50)")
    
    lr_row = results[results["Model"].str.contains("Logistic")]
    if not lr_row.empty:
        lr_val = lr_row["F1_Macro"].values[0]
        ax1.axhline(lr_val, color="grey", linestyle=":", label="LR Baseline")

    ax1.set_ylim(0.40, 0.60)
    ax1.set_ylabel("F1 Macro")
    ax1.set_title("F1 Score")
    ax1.legend(fontsize=8)

    # AUC Plot
    ax2.bar(finbert_data["Model"], finbert_data["AUC"], color=colors)
    ax2.axhline(0.50, color="red", linestyle="--", label="Random (0.50)")
    ax2.set_ylim(0.40, 0.60)
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    chart_out = os.path.join(RESULTS_DIR, "results_comparison_chart.png")
    plt.savefig(chart_out, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Chart saved to: {chart_out}")

if __name__ == "__main__":
    main()