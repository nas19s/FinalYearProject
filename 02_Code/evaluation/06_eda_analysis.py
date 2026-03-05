import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "eda")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Plotting defaults
sns.set_style("whitegrid")
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def main():
    # Load datasets
    df = pd.read_parquet(os.path.join(DATA_DIR, "labeled_dataset.parquet"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "sec_metadata.csv"))

    # Date normalization
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.tz_localize(None)
    df["year"] = df["filing_date"].dt.year

    meta["filing_date"] = pd.to_datetime(
        meta["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    meta["year"] = meta["filing_date"].dt.year

    print(f"Loaded {len(df):,} chunks and {len(meta):,} filings.")

    # Figure 1: Label distribution across horizons
    print("Generating Figure 1...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Label Distribution Across Prediction Horizons", fontsize=13, fontweight="bold")

    label_colors = {-1: "#F44336", 0: "#9E9E9E", 1: "#4CAF50"}
    label_names = {-1: "DOWN", 0: "FLAT", 1: "UP"}

    for ax, col, title in zip(
        axes,
        ["Label_T5", "Label_T10", "Label_T20"],
        ["T+5 (1 week)", "T+10 (2 weeks)", "T+20 (1 month)"]
    ):
        filing_labels = (df.groupby(["ticker", "filing_date"])[col]
                         .first()
                         .value_counts()
                         .sort_index())
        
        bar_colors = [label_colors.get(int(i), "#2196F3") for i in filing_labels.index]
        x_labels = [label_names.get(int(i), str(i)) for i in filing_labels.index]

        ax.bar(x_labels, filing_labels.values, color=bar_colors)
        ax.set_title(title)
        ax.set_xlabel("Label")
        ax.set_ylabel("Number of filings")
        for i, v in enumerate(filing_labels.values):
            ax.text(i, v + 5, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save(fig, "fig1_label_distributions.png")

    # Figure 2: Filing timeline
    print("Generating Figure 2...")
    fig, ax = plt.subplots(figsize=(12, 5))
    timeline = meta.groupby(["year", "form_type"]).size().unstack(fill_value=0)
    timeline.plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800"], rot=0)
    ax.set_title("SEC Filings by Year and Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of filings")
    ax.legend(title="Form type")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "fig2_filing_timeline.png")

    # Figure 3: Chunk length distribution
    print("Generating Figure 3...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["token_len"], bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(df["token_len"].mean(), color="#F44336", linestyle="--", linewidth=2,
               label=f"Mean = {df['token_len'].mean():.0f}")
    ax.axvline(512, color="#FF9800", linestyle="--", linewidth=2, label="Max (512)")
    ax.set_title("Token Length Distribution Across Chunks", fontsize=13, fontweight="bold")
    ax.set_xlabel("Token length")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    save(fig, "fig3_chunk_lengths.png")

    # Figure 4: Section coverage
    print("Generating Figure 4...")
    section_counts = df.groupby("section").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    section_counts.sort_values().plot(kind="barh", ax=ax, color=PALETTE[:len(section_counts)])
    ax.set_title("Chunks per SEC Filing Section", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of chunks")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width()):,}",
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    ha="left", va="center", fontsize=9, color="black")
    plt.tight_layout()
    save(fig, "fig4_section_coverage.png")

    # Figure 5: Label distribution by form type
    print("Generating Figure 5...")
    filing_df = df.groupby(["ticker", "filing_date", "form_type"])["Label_T20"].first().reset_index()
    filing_df["label_str"] = filing_df["Label_T20"].map({-1: "DOWN", 0: "FLAT", 1: "UP"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Label Distribution by Form Type (T+20)", fontsize=13, fontweight="bold")

    label_color_map = {"DOWN": "#F44336", "FLAT": "#9E9E9E", "UP": "#4CAF50"}

    for ax, form in zip(axes, ["10-K", "10-Q"]):
        subset = filing_df[filing_df["form_type"] == form]
        counts = subset["label_str"].value_counts().sort_index()
        bar_colors = [label_color_map.get(l, "#2196F3") for l in counts.index]
        ax.pie(counts.values, labels=counts.index, colors=bar_colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{form} ({len(subset):,} filings)")

    plt.tight_layout()
    save(fig, "fig5_labels_by_form_type.png")

    # Figure 6: Top tickers by filing count
    print("Generating Figure 6...")
    ticker_counts = meta.groupby("ticker").size().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    ticker_counts.plot(kind="bar", ax=ax, color="#2196F3", rot=45)
    ax.set_title("Top 20 Tickers by Filing Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Number of filings")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "fig6_top_tickers.png")

    print(f"Analysis complete. Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()