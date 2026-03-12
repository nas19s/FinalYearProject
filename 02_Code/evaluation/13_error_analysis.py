import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "error_analysis")
METRICS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)

# S&P 500 sector mapping
SECTOR_MAP = {
    "AAPL": "Technology",   "MSFT": "Technology",  "NVDA": "Technology",
    "GOOGL": "Technology",  "META": "Technology",   "AVGO": "Technology",
    "ORCL": "Technology",   "CRM": "Technology",    "AMD": "Technology",
    "INTC": "Technology",   "QCOM": "Technology",   "TXN": "Technology",
    "AMZN": "Consumer",     "TSLA": "Consumer",     "HD": "Consumer",
    "MCD": "Consumer",      "NKE": "Consumer",      "SBUX": "Consumer",
    "WMT": "Consumer",      "COST": "Consumer",      "TGT": "Consumer",
    "JPM": "Financials",    "BAC": "Financials",    "WFC": "Financials",
    "GS": "Financials",     "MS": "Financials",     "BRK-B": "Financials",
    "V": "Financials",      "MA": "Financials",     "AXP": "Financials",
    "JNJ": "Healthcare",    "UNH": "Healthcare",    "PFE": "Healthcare",
    "ABBV": "Healthcare",   "MRK": "Healthcare",    "ABT": "Healthcare",
    "LLY": "Healthcare",    "TMO": "Healthcare",
    "XOM": "Energy",        "CVX": "Energy",        "COP": "Energy",
    "NEE": "Utilities",     "DUK": "Utilities",
    "CAT": "Industrials",   "HON": "Industrials",   "UPS": "Industrials",
    "BA": "Industrials",    "RTX": "Industrials",
    "LIN": "Materials",     "APD": "Materials",
}

def main():
    print("Starting failure pattern analysis...")

    # Load predictions and classify errors
    preds = pd.read_csv(os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv"))
    preds["filing_date"] = pd.to_datetime(preds["filing_date"])
    preds["sector"] = preds["ticker"].map(SECTOR_MAP).fillna("Other")
    preds["correct"] = (preds["pred_label"] == preds["true_label"])
    
    preds["error_type"] = "Correct"
    preds.loc[(preds["pred_label"] == 1) & (preds["true_label"] == 0), "error_type"] = "False Positive"
    preds.loc[(preds["pred_label"] == 0) & (preds["true_label"] == 1), "error_type"] = "False Negative"
    preds.loc[(preds["pred_label"] == 0) & (preds["true_label"] == 0), "error_type"] = "True Negative"

    total = len(preds)
    n_correct = preds["correct"].sum()
    n_fp = (preds["error_type"] == "False Positive").sum()
    n_fn = (preds["error_type"] == "False Negative").sum()

    print(f"Stats: Total={total}, Accuracy={n_correct/total:.1%}, FP={n_fp}, FN={n_fn}")

    # Merge forward returns for magnitude analysis
    labels = pd.read_parquet(os.path.join(DATA_DIR, "labeled_dataset.parquet"))
    labels["filing_date"] = pd.to_datetime(labels["filing_date"]).dt.tz_localize(None)

    fwd = labels.groupby(["ticker", "filing_date"]).agg(fwd_return_T20=("fwd_return_T20", "first")).reset_index()
    preds["filing_date"] = pd.to_datetime(preds["filing_date"]).dt.tz_localize(None)
    fwd["filing_date"] = pd.to_datetime(fwd["filing_date"]).dt.tz_localize(None)
    preds = preds.merge(fwd, on=["ticker", "filing_date"], how="left")

    # Worst False Positives (Predicted UP, Fell Hardest)
    fp = preds[preds["error_type"] == "False Positive"].copy().sort_values("fwd_return_T20")
    worst_fp = fp.head(15)[["ticker", "filing_date", "sector", "confidence", "fwd_return_T20"]]
    worst_fp.to_csv(os.path.join(RESULTS_DIR, "worst_false_positives.csv"), index=False)

    # Worst False Negatives (Predicted DOWN, Rose Hardest)
    fn = preds[preds["error_type"] == "False Negative"].copy().sort_values("fwd_return_T20", ascending=False)
    worst_fn = fn.head(15)[["ticker", "filing_date", "sector", "confidence", "fwd_return_T20"]]
    worst_fn.to_csv(os.path.join(RESULTS_DIR, "worst_false_negatives.csv"), index=False)

    # Sector Stats
    sector_stats = (preds.groupby("sector")
                    .apply(lambda g: pd.Series({
                        "n_filings":  len(g),
                        "accuracy":   g["correct"].mean(),
                        "fp_rate":    (g["error_type"] == "False Positive").mean(),
                        "fn_rate":    (g["error_type"] == "False Negative").mean(),
                    }))
                    .reset_index().sort_values("accuracy", ascending=False))
    sector_stats.to_csv(os.path.join(RESULTS_DIR, "error_by_sector.csv"), index=False)

    # Temporal Stats
    preds["quarter"] = preds["filing_date"].dt.to_period("Q")
    quarter_stats = (preds.groupby("quarter")
                     .apply(lambda g: pd.Series({
                         "n_filings": len(g),
                         "accuracy":  g["correct"].mean(),
                         "fp_rate":   (g["error_type"] == "False Positive").mean(),
                     }))
                     .reset_index())
    quarter_stats["quarter"] = quarter_stats["quarter"].astype(str)
    quarter_stats.to_csv(os.path.join(RESULTS_DIR, "error_by_quarter.csv"), index=False)

    # Ticker Stats
    ticker_stats = (preds.groupby("ticker")
                    .apply(lambda g: pd.Series({
                        "n_filings": len(g),
                        "accuracy":  g["correct"].mean(),
                        "fp_count":  (g["error_type"] == "False Positive").sum(),
                        "fn_count":  (g["error_type"] == "False Negative").sum(),
                        "sector":    g["sector"].iloc[0],
                    }))
                    .reset_index().sort_values("accuracy"))
    ticker_stats.to_csv(os.path.join(RESULTS_DIR, "error_by_ticker.csv"), index=False)

    # Visualizations
    print("Generating analysis plots...")
    
    # Pie chart breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Error Analysis Summary (2023+ Test Set)", fontsize=13, fontweight="bold")
    
    counts = preds["error_type"].value_counts()
    colors = {"Correct": "#4CAF50", "False Positive": "#F44336", "False Negative": "#FF9800", "True Negative": "#2196F3"}
    axes[0].pie(counts.values, labels=counts.index, colors=[colors.get(x) for x in counts.index], autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Outcome Distribution")

    # Sector accuracy bar chart
    s_plot = sector_stats.sort_values("accuracy")
    b_colors = ["#4CAF50" if a >= 0.55 else "#FF9800" if a >= 0.50 else "#F44336" for a in s_plot["accuracy"]]
    axes[1].barh(s_plot["sector"], s_plot["accuracy"], color=b_colors)
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random")
    axes[1].axvline(preds["correct"].mean(), color="blue", linestyle="--", label="Mean")
    axes[1].set_title("Accuracy by Sector")
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_breakdown.png"), dpi=150)
    plt.close()

    # Quarterly accuracy
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(range(len(quarter_stats)), quarter_stats["accuracy"], color=["#4CAF50" if a >= 0.50 else "#F44336" for a in quarter_stats["accuracy"]])
    ax2.set_xticks(range(len(quarter_stats)))
    ax2.set_xticklabels(quarter_stats["quarter"], rotation=45)
    ax2.set_title("Quarterly Accuracy (Macro Stability Check)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_by_quarter.png"), dpi=150)
    plt.close()

    # Return distribution comparison
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    c_rets = preds[preds["correct"] & preds["fwd_return_T20"].notna()]["fwd_return_T20"]
    f_rets = preds[(preds["error_type"] == "False Positive") & preds["fwd_return_T20"].notna()]["fwd_return_T20"]
    ax3.hist(c_rets, bins=40, alpha=0.6, color="#4CAF50", label="Correct", edgecolor="white")
    ax3.hist(f_rets, bins=40, alpha=0.6, color="#F44336", label="False Positive", edgecolor="white")
    ax3.set_title("Return Distribution Comparison")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fp_return_distribution.png"), dpi=150)
    plt.close()

    print(f"Analysis complete. Results stored in {RESULTS_DIR}")

if __name__ == "__main__":
    main()