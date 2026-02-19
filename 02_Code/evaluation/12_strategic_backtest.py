"""
12_strategic_backtest.py — Event-Driven Backtest: FinBERT Risk-Ranked Portfolio
Each SEC filing = one independent trade. FinBERT reads the filing,
decides to BUY (hold 1 month) or SKIP. Compare average per-trade
returns against a naive "buy every filing" baseline.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
TENSOR_PATH  = os.path.join(PROJECT_ROOT, "03_Models", "finbert_tensors.pt")
CSV_PATH     = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "04_Results", "backtest")

TARGET_COL     = "Label_Month"
BATCH_SIZE     = 8
VAL_SPLIT      = 0.15
RANDOM_SEED    = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def trade_metrics(returns: np.ndarray, label: str) -> dict:
    """Compute per-trade statistics for a set of trade returns."""
    n = len(returns)
    if n == 0:
        return {
            "Strategy": label, "Trades": 0, "Mean Return %": 0,
            "Median Return %": 0, "Std %": 0, "Win Rate %": 0,
            "Avg Win %": 0, "Avg Loss %": 0, "Total Return %": 0,
            "Sharpe (per-trade)": 0, "Max Single Loss %": 0,
            "Max Single Win %": 0,
        }
    wins     = returns[returns > 0]
    losses   = returns[returns <= 0]
    win_rate = len(wins) / n * 100
    # Total return if you allocated equally to each trade
    total    = np.prod(1 + returns) - 1

    sharpe = 0.0
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(12)  # annualise monthly

    return {
        "Strategy":           label,
        "Trades":             n,
        "Mean Return %":      round(returns.mean() * 100, 2),
        "Median Return %":    round(np.median(returns) * 100, 2),
        "Std %":              round(returns.std() * 100, 2),
        "Win Rate %":         round(win_rate, 1),
        "Avg Win %":          round(wins.mean() * 100, 2) if len(wins) > 0 else 0,
        "Avg Loss %":         round(losses.mean() * 100, 2) if len(losses) > 0 else 0,
        "Total Return %":     round(total * 100, 2),
        "Sharpe (annualised)": round(sharpe, 3),
        "Max Single Loss %":  round(returns.min() * 100, 2),
        "Max Single Win %":   round(returns.max() * 100, 2),
    }


def main():
    print("=" * 65)
    print("  Event-Driven Backtest — FinBERT Risk-Ranked Portfolio")
    print("=" * 65)

    device = get_device()
    print(f"Device: {device}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    df       = pd.read_csv(CSV_PATH)
    df_clean = df[df[TARGET_COL] != -1].copy()

    # ── Load tensors and reconstruct val split ─────────────────────────────
    print("Loading tensors...")
    data      = torch.load(TENSOR_PATH, map_location="cpu")
    input_ids = data["input_ids"]
    attn_mask = data["attention_mask"]
    labels    = data["labels"]

    indices = list(range(len(labels)))
    _, val_idx = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels.numpy(),
    )

    val_idx_t  = torch.tensor(val_idx)
    val_ds     = TensorDataset(
        input_ids[val_idx_t], attn_mask[val_idx_t], labels[val_idx_t]
    )
    dataloader = DataLoader(
        val_ds, sampler=SequentialSampler(val_ds), batch_size=BATCH_SIZE
    )

    # Align CSV rows to val indices
    test_df = df_clean.iloc[val_idx].copy().reset_index(drop=True)
    test_df["join_date"] = pd.to_datetime(test_df["join_date"])

    # ── Load model & predict ───────────────────────────────────────────────
    print("Loading FinBERT...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    print("Generating P(Down) for each filing...\n")
    all_down_probs = []
    for batch in dataloader:
        ids, mask, _ = [t.to(device) for t in batch]
        with torch.no_grad():
            logits = model(ids, token_type_ids=None, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_down_probs.extend(probs[:, 0])

    test_df["Down_Prob"]    = all_down_probs
    test_df["Return_Month"] = test_df["Return_Month"].astype(float)

    # ── Baseline: Buy every filing ─────────────────────────────────────────
    all_returns = test_df["Return_Month"].values
    baseline    = trade_metrics(all_returns, "Buy All Filings (Baseline)")

    # ── Strategy variants ──────────────────────────────────────────────────
    print("Running strategies across thresholds...\n")
    results = [baseline]

    # Long-only: skip risky filings
    for skip_pct in [10, 15, 20, 25]:
        risk_pct  = 100 - skip_pct
        threshold = np.percentile(all_down_probs, risk_pct)
        mask_safe = test_df["Down_Prob"] < threshold
        safe_ret  = test_df.loc[mask_safe, "Return_Month"].values
        label     = f"Skip Top {skip_pct}% Risk (Long Only)"
        results.append(trade_metrics(safe_ret, label))

    # Long-short: buy safe, short risky
    for skip_pct in [10, 15, 20]:
        risk_pct  = 100 - skip_pct
        threshold = np.percentile(all_down_probs, risk_pct)
        safe_ret  = test_df.loc[test_df["Down_Prob"] < threshold, "Return_Month"].values
        risky_ret = test_df.loc[test_df["Down_Prob"] >= threshold, "Return_Month"].values
        # Short returns: profit when stock goes down
        short_ret = -risky_ret
        combined  = np.concatenate([safe_ret, short_ret])
        label     = f"Long Safe / Short Top {skip_pct}% (L/S)"
        results.append(trade_metrics(combined, label))

    # ── Results table ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results)

    print("=" * 100)
    print("BACKTEST RESULTS — PER-TRADE ANALYSIS")
    print("=" * 100)
    # Display key columns
    display_cols = [
        "Strategy", "Trades", "Mean Return %", "Win Rate %",
        "Sharpe (annualised)", "Total Return %", "Max Single Loss %"
    ]
    print(summary_df[display_cols].to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, "backtest_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nFull summary saved → {csv_path}")

    # ── Plot 1: Per-trade return distributions ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Histogram of P(Down) scores
    ax = axes[0, 0]
    ax.hist(all_down_probs, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    for pct in [80, 85, 90]:
        thresh = np.percentile(all_down_probs, pct)
        ax.axvline(thresh, color="red", linestyle="--", alpha=0.6,
                   label=f"Top {100-pct}% threshold: {thresh:.3f}")
    ax.set_title("Distribution of FinBERT P(Down) Scores", fontsize=12)
    ax.set_xlabel("P(Down)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Top-right: Box plot comparing trade returns
    ax = axes[0, 1]
    threshold_85 = np.percentile(all_down_probs, 85)
    safe_mask    = test_df["Down_Prob"] < threshold_85
    safe_rets    = test_df.loc[safe_mask, "Return_Month"].values * 100
    risky_rets   = test_df.loc[~safe_mask, "Return_Month"].values * 100
    bp = ax.boxplot(
        [safe_rets, risky_rets, all_returns * 100],
        labels=["Safe (Bottom 85%)", "Risky (Top 15%)", "All Filings"],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
    )
    colours = ["#2ecc71", "#e74c3c", "#3498db"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.5)
    ax.set_title("Trade Return Distribution by Risk Group", fontsize=12)
    ax.set_ylabel("1-Month Return (%)")
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Mean return comparison bar chart
    ax = axes[1, 0]
    long_only_results = [r for r in results if "Long Only" in r["Strategy"] or "Baseline" in r["Strategy"]]
    names   = [r["Strategy"].replace(" (Long Only)", "").replace(" (Baseline)", "")
               for r in long_only_results]
    means   = [r["Mean Return %"] for r in long_only_results]
    bar_colors = ["#3498db"] + ["#2ecc71"] * (len(means) - 1)
    bars = ax.bar(range(len(names)), means, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_title("Mean Per-Trade Return by Strategy", fontsize=12)
    ax.set_ylabel("Mean Return (%)")
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Bottom-right: Sharpe ratio comparison
    ax = axes[1, 1]
    all_strat_names  = [r["Strategy"].split("(")[0].strip() for r in results]
    all_strat_sharpe = [r["Sharpe (annualised)"] for r in results]
    bar_colors = []
    for r in results:
        if "Baseline" in r["Strategy"]:
            bar_colors.append("#3498db")
        elif "L/S" in r["Strategy"]:
            bar_colors.append("#9b59b6")
        else:
            bar_colors.append("#2ecc71")
    bars = ax.barh(range(len(all_strat_names)), all_strat_sharpe,
                   color=bar_colors, edgecolor="white")
    ax.set_yticks(range(len(all_strat_names)))
    ax.set_yticklabels(all_strat_names, fontsize=8)
    ax.set_title("Annualised Sharpe Ratio by Strategy", fontsize=12)
    ax.set_xlabel("Sharpe Ratio")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, all_strat_sharpe):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", ha="left", va="center", fontsize=9)

    plt.suptitle("FinBERT Event-Driven Backtest Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "equity_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {plot_path}")

    # ── Interpretation ─────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["Sharpe (annualised)"])
    print(f"\n{'=' * 65}")
    print(f"  Best strategy by Sharpe: {best['Strategy']}")
    print(f"  Sharpe: {best['Sharpe (annualised)']:.3f}  |  "
          f"Mean Return: {best['Mean Return %']:.2f}%  |  "
          f"Win Rate: {best['Win Rate %']:.1f}%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()