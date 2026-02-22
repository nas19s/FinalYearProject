"""
13_error_analysis.py — Worst Prediction Analysis
Extracts the most confident wrong predictions and explains why.
Saves results to 04_Results/error_analysis/
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification

# ── Paths (matched to your project structure) ─────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
TENSOR_PATH  = os.path.join(PROJECT_ROOT, "03_Models", "finbert_tensors.pt")
CSV_PATH     = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "04_Results", "error_analysis")

TARGET_COL   = "Label_Month"
BATCH_SIZE   = 8
VAL_SPLIT    = 0.15
RANDOM_SEED  = 42
TOP_N        = 10

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    print("=" * 65)
    print("  Error Analysis — Worst FinBERT Predictions")
    print("=" * 65)

    device = get_device()
    print(f"Device: {device}\n")

    # ── Load CSV ───────────────────────────────────────────────────────
    print(f"Loading CSV from {CSV_PATH}")
    df       = pd.read_csv(CSV_PATH)
    df_clean = df[df[TARGET_COL] != -1].copy()
    print(f"  Clean rows: {len(df_clean)}")

    # ── Check which columns actually exist ─────────────────────────────
    available_cols = df_clean.columns.tolist()
    print(f"  Available columns: {len(available_cols)}")

    # Map expected column names to possible variants in your dataset
    col_map = {}
    for candidate in ["Ticker", "ticker", "Symbol", "symbol"]:
        if candidate in available_cols:
            col_map["ticker"] = candidate
            break

    for candidate in ["join_date", "Date", "date", "filing_date"]:
        if candidate in available_cols:
            col_map["date"] = candidate
            break

    for candidate in ["Return_Month", "return_month"]:
        if candidate in available_cols:
            col_map["return"] = candidate
            break

    for candidate in ["RSI_14", "rsi_14", "RSI"]:
        if candidate in available_cols:
            col_map["rsi"] = candidate
            break

    for candidate in ["Gunning_Fog", "gunning_fog"]:
        if candidate in available_cols:
            col_map["fog"] = candidate
            break

    for candidate in ["Flesch_Reading_Ease", "flesch_reading_ease"]:
        if candidate in available_cols:
            col_map["flesch"] = candidate
            break

    for candidate in ["vader_compound", "VADER_Compound", "Vader_Compound"]:
        if candidate in available_cols:
            col_map["vader"] = candidate
            break

    for candidate in ["MACD", "macd"]:
        if candidate in available_cols:
            col_map["macd"] = candidate
            break

    print(f"  Mapped columns: {col_map}\n")

    # ── Load tensors and reconstruct val split ─────────────────────────
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
    print(f"Validation samples: {len(test_df)}")

    # ── Load model & predict ───────────────────────────────────────────
    print("Loading FinBERT champion model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    print("Generating predictions...\n")
    all_probs = []
    all_preds = []
    all_true  = []

    for batch in dataloader:
        ids, mask, labs = [t.to(device) for t in batch]
        with torch.no_grad():
            logits = model(ids, token_type_ids=None, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(labs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    label_map = {0: "Down", 1: "Up"}

    # ── Overall accuracy breakdown ─────────────────────────────────────
    correct   = (all_preds == all_true).sum()
    total     = len(all_true)
    wrong_mask = all_preds != all_true
    wrong_idx  = np.where(wrong_mask)[0]

    print(f"Overall: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Wrong:   {len(wrong_idx)}/{total} ({len(wrong_idx)/total*100:.1f}%)")

    # ── Find worst errors (most confident AND wrong) ───────────────────
    wrong_conf = np.array([
        all_probs[i, all_preds[i]] for i in wrong_idx
    ])

    sorted_order = np.argsort(-wrong_conf)
    top_wrong    = wrong_idx[sorted_order[:TOP_N]]

    # Helper to safely get column value
    def safe_get(row, key, default=np.nan):
        col = col_map.get(key)
        if col and col in row.index:
            val = row[col]
            try:
                return float(val)
            except (ValueError, TypeError):
                return val
        return default

    print(f"\n{'=' * 95}")
    print(f"TOP {TOP_N} MOST CONFIDENT WRONG PREDICTIONS")
    print(f"{'=' * 95}")
    print(f"{'#':>3} {'Ticker':<8} {'Date':<12} {'True':<6} {'Pred':<6} "
          f"{'Conf%':>6} {'Return%':>9} {'RSI':>6} {'Fog':>6} {'Error Type'}")
    print("-" * 95)

    error_rows = []
    for rank, idx in enumerate(top_wrong, 1):
        row  = test_df.iloc[idx]
        true = label_map[all_true[idx]]
        pred = label_map[all_preds[idx]]
        conf = all_probs[idx, all_preds[idx]] * 100
        ret  = safe_get(row, "return", np.nan)
        if not np.isnan(ret):
            ret_pct = ret * 100
        else:
            ret_pct = np.nan

        ticker = safe_get(row, "ticker", "?")
        if isinstance(ticker, float):
            ticker = "?"
        date   = str(safe_get(row, "date", "?"))[:10]
        rsi    = safe_get(row, "rsi")
        fog    = safe_get(row, "fog")
        flesch = safe_get(row, "flesch")
        vader  = safe_get(row, "vader")
        macd   = safe_get(row, "macd")

        if true == "Down" and pred == "Up":
            error_type = "FALSE POSITIVE (missed crash)"
        else:
            error_type = "FALSE NEGATIVE (missed rally)"

        rsi_str = f"{rsi:>6.1f}" if not np.isnan(rsi) else "   N/A"
        fog_str = f"{fog:>6.1f}" if not np.isnan(fog) else "   N/A"
        ret_str = f"{ret_pct:>8.1f}%" if not np.isnan(ret_pct) else "     N/A"

        print(f"{rank:>3} {str(ticker):<8} {date:<12} {true:<6} {pred:<6} "
              f"{conf:>5.1f}% {ret_str} {rsi_str} {fog_str} {error_type}")

        error_rows.append({
            "Rank":               rank,
            "Ticker":             ticker,
            "Date":               date,
            "True_Label":         true,
            "Predicted":          pred,
            "Confidence%":        round(conf, 1),
            "P_Down":             round(all_probs[idx, 0], 4),
            "P_Up":               round(all_probs[idx, 1], 4),
            "Return%":            round(ret_pct, 1) if not np.isnan(ret_pct) else "",
            "RSI_14":             round(rsi, 1) if not np.isnan(rsi) else "",
            "MACD":               round(macd, 4) if not np.isnan(macd) else "",
            "Gunning_Fog":        round(fog, 1) if not np.isnan(fog) else "",
            "Flesch_Reading_Ease": round(flesch, 1) if not np.isnan(flesch) else "",
            "VADER_Compound":     round(vader, 4) if not np.isnan(vader) else "",
            "Error_Type":         error_type,
        })

    # ── Save errors CSV ────────────────────────────────────────────────
    error_df = pd.DataFrame(error_rows)
    csv_path = os.path.join(RESULTS_DIR, "worst_predictions.csv")
    error_df.to_csv(csv_path, index=False)

    # ── Error pattern analysis ─────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("ERROR PATTERN ANALYSIS")
    print(f"{'=' * 65}")

    fp_count = sum(1 for r in error_rows if "FALSE POSITIVE" in r["Error_Type"])
    fn_count = sum(1 for r in error_rows if "FALSE NEGATIVE" in r["Error_Type"])
    print(f"\n  Error type breakdown (top {TOP_N}):")
    print(f"    False Positives (missed crashes): {fp_count}")
    print(f"    False Negatives (missed rallies): {fn_count}")

    if fp_count > fn_count:
        print("    → Model is OVERCONFIDENTLY BULLISH — it misses downturns.")
    elif fn_count > fp_count:
        print("    → Model is OVERCONFIDENTLY BEARISH — it misses rallies.")
    else:
        print("    → Errors are balanced between both types.")

    # Compare features in errors vs overall validation set
    print(f"\n  Feature comparison (errors vs overall validation):")
    print(f"  {'Metric':<25} {'Error Avg':>12} {'Overall Avg':>12} {'Signal'}")
    print(f"  {'-'*65}")

    comparisons = [
        ("vader", "VADER Compound"),
        ("fog",   "Gunning Fog"),
        ("flesch", "Flesch Reading Ease"),
        ("rsi",   "RSI-14"),
    ]

    insights = []
    for key, name in comparisons:
        col = col_map.get(key)
        if col is None:
            continue

        error_vals = [
            r.get(
                {"vader": "VADER_Compound", "fog": "Gunning_Fog",
                 "flesch": "Flesch_Reading_Ease", "rsi": "RSI_14"}[key], ""
            )
            for r in error_rows
        ]
        error_vals = [v for v in error_vals if v != "" and not np.isnan(float(v))]

        if not error_vals:
            continue

        error_avg   = np.mean([float(v) for v in error_vals])
        overall_avg = test_df[col].mean()

        diff = error_avg - overall_avg
        if abs(diff) > 0.01 * abs(overall_avg) if overall_avg != 0 else abs(diff) > 0.01:
            direction = "↑ HIGHER" if diff > 0 else "↓ LOWER"
        else:
            direction = "≈ SIMILAR"

        print(f"  {name:<25} {error_avg:>12.4f} {overall_avg:>12.4f} {direction}")

        if key == "vader" and abs(error_avg) < abs(overall_avg):
            insights.append(
                "Errors cluster around NEUTRAL sentiment — the model "
                "struggles when text tone is genuinely ambiguous."
            )
        if key == "fog" and error_avg > overall_avg:
            insights.append(
                "Errors involve MORE COMPLEX text (higher Gunning Fog) — "
                "dense regulatory language reduces FinBERT's accuracy."
            )
        if key == "rsi" and 40 < error_avg < 60:
            insights.append(
                "Errors occur when RSI is near neutral (40-60), suggesting "
                "the model fails when there is no strong technical signal to lean on."
            )

    # ── Key insights ───────────────────────────────────────────────────
    if insights:
        print(f"\n  KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"    {i}. {insight}")

    # ── Confidence distribution plot ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Confidence distribution — correct vs wrong
    ax = axes[0, 0]
    correct_conf = [all_probs[i, all_preds[i]] for i in range(total) if all_preds[i] == all_true[i]]
    wrong_conf_all = [all_probs[i, all_preds[i]] for i in wrong_idx]
    ax.hist(correct_conf, bins=20, alpha=0.6, color="green", label="Correct", edgecolor="white")
    ax.hist(wrong_conf_all, bins=20, alpha=0.6, color="red", label="Wrong", edgecolor="white")
    ax.set_title("Prediction Confidence: Correct vs Wrong", fontsize=12)
    ax.set_xlabel("Confidence in Predicted Class")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Error type breakdown
    ax = axes[0, 1]
    all_fp = sum(1 for i in wrong_idx if all_true[i] == 0 and all_preds[i] == 1)
    all_fn = sum(1 for i in wrong_idx if all_true[i] == 1 and all_preds[i] == 0)
    bars = ax.bar(
        ["False Positive\n(Missed Crash)", "False Negative\n(Missed Rally)"],
        [all_fp, all_fn],
        color=["#e74c3c", "#3498db"],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, [all_fp, all_fn]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=14, fontweight="bold",
        )
    ax.set_title(f"All Errors Breakdown ({len(wrong_idx)} total)", fontsize=12)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: VADER compound in errors vs correct
    vader_col = col_map.get("vader")
    ax = axes[1, 0]
    if vader_col:
        correct_idx   = np.where(~wrong_mask)[0]
        correct_vader = test_df.iloc[correct_idx][vader_col].dropna().values
        wrong_vader   = test_df.iloc[wrong_idx][vader_col].dropna().values
        bp = ax.boxplot(
            [correct_vader, wrong_vader],
            labels=["Correct Predictions", "Wrong Predictions"],
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
        )
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#e74c3c")
        bp["boxes"][1].set_alpha(0.5)
        ax.set_title("VADER Sentiment: Correct vs Wrong", fontsize=12)
        ax.set_ylabel("VADER Compound Score")
        ax.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "VADER column not found", ha="center", va="center")
        ax.set_title("VADER Sentiment (unavailable)")

    # Plot 4: Return magnitude in errors
    return_col = col_map.get("return")
    ax = axes[1, 1]
    if return_col:
        correct_ret = test_df.iloc[np.where(~wrong_mask)[0]][return_col].dropna().values * 100
        wrong_ret   = test_df.iloc[wrong_idx][return_col].dropna().values * 100
        bp = ax.boxplot(
            [correct_ret, wrong_ret],
            labels=["Correct Predictions", "Wrong Predictions"],
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
        )
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#e74c3c")
        bp["boxes"][1].set_alpha(0.5)
        ax.set_title("Actual Returns: Correct vs Wrong", fontsize=12)
        ax.set_ylabel("1-Month Return (%)")
        ax.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Return column not found", ha="center", va="center")
        ax.set_title("Returns (unavailable)")

    plt.suptitle(
        "FinBERT Error Analysis — Where the Model Fails",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "error_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("FILES SAVED")
    print(f"{'=' * 65}")
    print(f"  CSV:  {csv_path}")
    print(f"  Plot: {plot_path}")
    print(f"\nPaste this into your report:")
    print(f"-" * 65)

    report_text = (
        f"Error analysis of the {len(wrong_idx)} misclassified validation "
        f"samples ({len(wrong_idx)/total*100:.1f}% error rate) reveals that "
        f"{fp_count} of the top {TOP_N} most confident errors were false "
        f"positives (the model predicted Up but the stock declined). "
    )
    if insights:
        report_text += " ".join(insights)
    report_text += (
        " These findings suggest that model performance could be improved "
        "through domain-specific pre-training on SEC disclosure language "
        "and the incorporation of stronger technical indicators for "
        "ambiguous-sentiment filings."
    )
    print(f"\n  {report_text}")
    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()