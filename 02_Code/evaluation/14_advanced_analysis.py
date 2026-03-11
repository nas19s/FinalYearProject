import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                              ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, '01_Data')
PRICES_DIR = os.path.join(DATA_DIR, 'prices')
RESULTS_DIR = os.path.join(PROJECT_ROOT, '04_Results', 'advanced')
METRICS_DIR = os.path.join(PROJECT_ROOT, '04_Results', 'metrics')
os.makedirs(RESULTS_DIR, exist_ok=True)

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Technology","META":"Technology","AVGO":"Technology",
    "ORCL":"Technology","CRM":"Technology","AMD":"Technology",
    "INTC":"Technology","QCOM":"Technology","TXN":"Technology",
    "AMZN":"Consumer","TSLA":"Consumer","HD":"Consumer",
    "MCD":"Consumer","NKE":"Consumer","SBUX":"Consumer",
    "WMT":"Consumer","COST":"Consumer","TGT":"Consumer",
    "JPM":"Financials","BAC":"Financials","WFC":"Financials",
    "GS":"Financials","MS":"Financials","BRK-B":"Financials",
    "V":"Financials","MA":"Financials","AXP":"Financials",
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare",
    "ABBV":"Healthcare","MRK":"Healthcare","ABT":"Healthcare",
    "LLY":"Healthcare","TMO":"Healthcare",
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "NEE":"Utilities","DUK":"Utilities",
    "CAT":"Industrials","HON":"Industrials","UPS":"Industrials",
    "BA":"Industrials","RTX":"Industrials",
    "LIN":"Materials","APD":"Materials",
}

def load_prices(ticker):
    for suffix in ("_prices.csv", ".csv"):
        path = os.path.join(PRICES_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                df = df.sort_values("Date").set_index("Date")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
            except Exception:
                return None
    return None

def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Exported: {name}")

def plot_calibration():
    print('Analyzing model calibration...')
    path = os.path.join(METRICS_DIR, "finbert_test_predictions_T20.csv")
    if not os.path.exists(path): return

    df = pd.read_csv(path)
    y_true = df["true_label"].astype(int).values
    y_prob = df["prob_up"].astype(float).values

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("FinBERT T+20 Calibration Analysis", fontsize=13, fontweight="bold")

    axes[0].plot([0,1],[0,1],"k--", label="Ideal")
    axes[0].plot(mean_pred, frac_pos, "s-", color="#2196F3", label="FinBERT T20")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Observed Fraction")
    axes[0].set_title("Reliability Diagram")
    axes[0].legend()

    axes[1].hist(y_prob[y_true==0], bins=30, alpha=0.6, color="#F44336", label="True DOWN")
    axes[1].hist(y_prob[y_true==1], bins=30, alpha=0.6, color="#4CAF50", label="True UP")
    axes[1].set_title("Probability Distribution")
    axes[1].legend()

    plt.tight_layout()
    save(fig, "A_calibration_plot.png")
    

def plot_caar_event_study():
    print("Executing CAAR Event Study...")
    labels_path = os.path.join(DATA_DIR, "labeled_dataset.parquet")
    if not os.path.exists(labels_path): return
    
    labels = pd.read_parquet(labels_path)
    labels["filing_date"] = pd.to_datetime(labels["filing_date"]).dt.tz_localize(None)
    labels = labels[labels["filing_date"] >= "2016-01-01"]

    filings = (labels.groupby(["ticker", "filing_date"])
               .agg(Label_T20=("Label_T20", "first")).reset_index())

    event_window = range(-10, 21)
    rets_up, rets_down = {d: [] for d in event_window}, {d: [] for d in event_window}

    for _, row in filings.iterrows():
        prices = load_prices(row["ticker"])
        if prices is None: continue
        
        loc = prices.index.searchsorted(row["filing_date"])
        if loc < 10 or loc + 21 > len(prices.index): continue

        col = next((c for c in ["Close", "Adj Close"] if c in prices.columns), None)
        if not col: continue

        window = prices[col].iloc[loc - 10: loc + 21]
        base = window.iloc[10]
        if pd.isna(base) or base <= 0: continue

        norm = (window.values / base - 1)
        for i, d in enumerate(event_window):
            if row["Label_T20"] == 1: rets_up[d].append(norm[i])
            elif row["Label_T20"] == -1: rets_down[d].append(norm[i])

    days = list(event_window)
    caar_up = [np.mean(rets_up[d]) if rets_up[d] else 0 for d in days]
    caar_down = [np.mean(rets_down[d]) if rets_down[d] else 0 for d in days]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, [r*100 for r in caar_up], color="#4CAF50", label="UP Filings", lw=2)
    ax.plot(days, [r*100 for r in caar_down], color="#F44336", label="DOWN Filings", lw=2)
    ax.axvline(0, color="black", linestyle="--", label="Release Day")
    ax.set_title("CAAR Event Study: Cumulative Abnormal Returns")
    ax.set_ylabel("CAR (%)")
    ax.legend()
    save(fig, "B_caar_event_study.png")
    

def plot_sector_breakdown():
    print('Calculating sector performance...')
    preds_path = os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv")
    if not os.path.exists(preds_path): return
    
    preds = pd.read_csv(preds_path)
    preds["sector"] = preds["ticker"].map(SECTOR_MAP).fillna("Other")
    preds["correct"] = (preds["pred_label"] == preds["true_label"])

    sector_stats = (preds.groupby("sector")["correct"].mean()
                    .reset_index().sort_values("correct", ascending=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sector_stats["sector"], sector_stats["correct"]*100, color="#2196F3")
    ax.axhline(50, color="red", linestyle="--", label="Random")
    ax.set_title("Accuracy Across Market Sectors")
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    save(fig, "C_sector_breakdown.png")

def plot_confidence_sensitivity():
    print('Testing confidence threshold sensitivity...')
    preds = pd.read_csv(os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv"))
    if "confidence" not in preds.columns: return

    thresholds = np.arange(0.50, 0.91, 0.02)
    results = []
    for t in thresholds:
        subset = preds[preds["confidence"] >= t]
        if len(subset) < 10: break
        results.append({"threshold": t, "accuracy": (subset["pred_label"] == subset["true_label"]).mean()})

    res_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res_df["threshold"], res_df["accuracy"]*100, "o-", lw=2)
    ax.set_title("Accuracy Improvement via Confidence Gating")
    ax.set_xlabel("Minimum Confidence")
    ax.set_ylabel("Accuracy (%)")
    save(fig, "D_confidence_sensitivity.png")

def plot_roc_curves():
    print('Plotting ROC curves...')
    fig, ax = plt.subplots(figsize=(8, 7))
    for h in ["T5", "T10", "T20"]:
        path = os.path.join(METRICS_DIR, f"finbert_test_predictions_{h}.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        fpr, tpr, _ = roc_curve(df["true_label"].astype(int), df["prob_up"].astype(float))
        ax.plot(fpr, tpr, label=f"{h} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0,1], [0,1], "k--")
    ax.set_title("ROC Comparison Across Horizons")
    ax.legend()
    save(fig, "E_roc_curves.png")
    

def plot_length_vs_accuracy():
    print('Analyzing filing length vs accuracy...')
    preds = pd.read_csv(os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv"))
    preds["filing_date"] = pd.to_datetime(preds["filing_date"]).dt.tz_localize(None)
    preds["correct"] = (preds["pred_label"] == preds["true_label"])

    labels = pd.read_parquet(os.path.join(DATA_DIR, "labeled_dataset.parquet"))
    labels["filing_date"] = pd.to_datetime(labels["filing_date"]).dt.tz_localize(None)

    lengths = labels.groupby(["ticker", "filing_date"]).size().reset_index(name="n_chunks")
    merged = preds.merge(lengths, on=["ticker", "filing_date"], how="inner")

    if merged.empty or "n_chunks" not in merged.columns: return

    merged["bin"] = pd.cut(merged["n_chunks"], bins=[0, 20, 50, 100, 200, 1000], 
                           labels=["1-20", "21-50", "51-100", "101-200", "200+"])
    acc_by_len = merged.groupby("bin")["correct"].agg(['mean', 'count'])

    fig, ax = plt.subplots(figsize=(10, 5))
    acc_by_len['mean'].plot(kind="bar", ax=ax, color="#4CAF50")
    ax.set_title("Impact of Filing Complexity on Accuracy")
    ax.set_ylabel("Accuracy")
    save(fig, "F_length_vs_accuracy.png")

def run_significance_tests():
    print('Performing significance tests...')
    df = pd.read_csv(os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv"))
    n_correct = (df["pred_label"] == df["true_label"]).sum()
    n_total = len(df)
    
    binom = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
    print(f"Accuracy vs Random p-value: {binom.pvalue:.4f}")

def plot_finbert_confusion_matrices():
    print('Generating confusion matrices...')
    horizons = ["T5", "T10", "T20"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, h in enumerate(horizons):
        path = os.path.join(METRICS_DIR, f"finbert_test_predictions_{h}.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        cm = confusion_matrix(df["true_label"].astype(int), (df["prob_up"] >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["DOWN", "UP"]).plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(f"Horizon {h}")
    plt.tight_layout()
    save(fig, "I_finbert_confusion_matrices.png")
    

def main():
    plot_calibration()
    plot_caar_event_study()
    plot_sector_breakdown()
    plot_confidence_sensitivity()
    plot_roc_curves()
    plot_length_vs_accuracy()
    run_significance_tests()
    plot_finbert_confusion_matrices()
    print("All advanced analysis completed successfully.")

if __name__ == '__main__':
    main()