import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# Project Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "shap")
METRICS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "metrics")
MODELS_DIR = os.path.join(PROJECT_ROOT, "03_Models", "hybrid_ensemble")

os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_COL = "Label_T20"

# Mapping for dissertation-ready labels
FEATURE_LABELS = {
    "RSI": "RSI (Momentum)",
    "MACD": "MACD (Trend)",
    "Volume_Change": "Volume Change",
    "Gunning_Fog": "Gunning Fog (Readability)",
    "Flesch_Ease": "Flesch Reading Ease",
    "Sentiment": "VADER Sentiment",
    "Diff_Word_Ratio": "Difficult Word Ratio",
    "Word_Count": "Word Count",
    "confidence": "FinBERT Voting Confidence",
}

def main():
    print("Starting SHAP Explainability Analysis...")

    # Load artifacts
    try:
        with open(os.path.join(MODELS_DIR, "hybrid_rf_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "hybrid_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "hybrid_feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        return

    # Build the test set for analysis
    df = pd.read_parquet(os.path.join(DATA_DIR, "final_feature_dataset.parquet"))
    df = df[df[TARGET_COL] != 0].copy()
    df["binary_label"] = df[TARGET_COL].map({-1: 0, 1: 1})
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    base_cols = [c for c in feature_cols if c != "confidence"]
    df = df.dropna(subset=["binary_label"] + base_cols)

    # Use 2023+ data for testing
    test_df = df[df["filing_date"] >= "2023-01-01"].copy()
    agg_dict = {col: "mean" for col in base_cols}
    agg_dict["binary_label"] = "first"
    test_agg = test_df.groupby(["ticker", "filing_date"]).agg(agg_dict).reset_index()

    # Integrate FinBERT confidence scores
    if "confidence" in feature_cols:
        voting_path = os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv")
        if os.path.exists(voting_path):
            voting = pd.read_csv(voting_path)
            voting["filing_date"] = pd.to_datetime(voting["filing_date"])
            test_agg = test_agg.merge(
                voting[["ticker", "filing_date", "confidence"]],
                on=["ticker", "filing_date"], how="left"
            )
            test_agg["confidence"] = test_agg["confidence"].fillna(0.5)

    X_test = test_agg[feature_cols].fillna(0)
    X_test_s = scaler.transform(X_test)

    # Prepare DataFrame with readable feature names
    readable_names = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    X_named = pd.DataFrame(X_test_s, columns=readable_names)

    # Calculate SHAP values
    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_named)

    # Handle binary output structure
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Plot 1: Beeswarm Summary
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_named, plot_type="dot", show=False, max_display=len(readable_names))
    plt.title("SHAP Feature Impact: Predicting Price Direction", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Average Importance Bar Chart
    mean_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": readable_names,
        "Mean_SHAP": mean_shap,
    }).sort_values("Mean_SHAP", ascending=True)

    bar_colors = ["#FF9800" if "FinBERT" in f else "#2196F3" for f in shap_df["Feature"]]

    plt.figure(figsize=(10, 6))
    plt.barh(shap_df["Feature"], shap_df["Mean_SHAP"], color=bar_colors)
    plt.xlabel("Mean |SHAP Value| (Impact Strength)")
    plt.title("Feature Importance Ranking (Hybrid Model)", fontsize=12)

    legend_elements = [
        Patch(color="#FF9800", label="FinBERT Models"),
        Patch(color="#2196F3", label="Technical/NLP Metrics")
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Export values for documentation
    pd.DataFrame(sv, columns=readable_names).to_csv(
        os.path.join(RESULTS_DIR, "shap_values.csv"), index=False
    )

    # Ranking Printout
    shap_ranked = pd.DataFrame({
        "Feature": readable_names,
        "Mean_SHAP": mean_shap,
    }).sort_values("Mean_SHAP", ascending=False)

    print("\nFeature Ranking by SHAP Influence:")
    print(shap_ranked.to_string(index=False))
    print(f"\nAnalysis complete. Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()