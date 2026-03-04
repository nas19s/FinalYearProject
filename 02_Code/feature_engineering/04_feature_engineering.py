"""
04_feature_engineering.py

"""

import os
import re
import pandas as pd
import numpy as np
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
PRICES_DIR   = os.path.join(DATA_DIR, "prices")
INPUT_FILE   = os.path.join(DATA_DIR, "labeled_dataset.parquet")
OUTPUT_FILE  = os.path.join(DATA_DIR, "final_feature_dataset.parquet")

ANALYSIS_LIMIT = 100_000
analyzer = SentimentIntensityAnalyzer()

# ──────────────────────────────────────────────────────────────────────────────
_RE_HTML       = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")


def compute_nlp_features(text: str) -> dict:
    """Compute readability and sentiment features from a text chunk."""
    if not text or len(text.strip()) < 100:
        return {"Gunning_Fog": 0, "Flesch_Ease": 0,
                "Sentiment": 0, "Diff_Word_Ratio": 0, "Word_Count": 0}
    try:
        prose = _RE_HTML.sub(" ", text)
        prose = prose.replace("\xa0", " ")
        prose = _RE_WHITESPACE.sub(" ", prose).strip()
        prose = prose[:ANALYSIS_LIMIT]

        fog     = textstat.gunning_fog(prose)
        flesch  = textstat.flesch_reading_ease(prose)
        vs      = analyzer.polarity_scores(prose)
        words   = prose.split()
        diff_r  = textstat.difficult_words(prose) / (len(words) + 1)

        return {
            "Gunning_Fog":    fog,
            "Flesch_Ease":    flesch,
            "Sentiment":      vs["compound"],
            "Diff_Word_Ratio": diff_r,
            "Word_Count":     len(words),
        }
    except Exception:
        return {"Gunning_Fog": 0, "Flesch_Ease": 0,
                "Sentiment": 0, "Diff_Word_Ratio": 0, "Word_Count": 0}


def get_technical_indicators(ticker: str) -> pd.DataFrame | None:
    """Load price data and compute RSI, MACD, Volume Change."""
    for suffix in ("_prices.csv", ".csv"):
        path = os.path.join(PRICES_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                df = df.sort_values("Date").drop_duplicates("Date")
                df["RSI"]           = RSIIndicator(close=df["Close"]).rsi()
                df["MACD"]          = MACD(close=df["Close"]).macd()
                df["Volume_Change"] = df["Volume"].pct_change()
                return df[["Date", "RSI", "MACD", "Volume_Change"]]
            except Exception as e:
                print(f"  [{ticker}] Technical indicator error: {e}")
                return None
    return None


def main():
    print("=" * 70)
    print("04_feature_engineering.py")
    print("=" * 70)

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} rows from labeled_dataset.parquet")

    # ── 1. NLP features — computed per chunk ──────────────────────────────
    # Use the already-cleaned text column from sec_cleaned.parquet
    print("\nComputing NLP features (Fog, Sentiment etc.)...")
    nlp_rows = []
    for text in tqdm(df["text"], desc="NLP features"):
        nlp_rows.append(compute_nlp_features(str(text)))

    nlp_df = pd.DataFrame(nlp_rows, index=df.index)
    df = pd.concat([df, nlp_df], axis=1)

    # ── 2. Technical indicators — computed per ticker, merged on entry_date ─
    print("\nComputing technical indicators...")
    indicator_frames = []
    for ticker in tqdm(df["ticker"].unique(), desc="Technical indicators"):
        tech = get_technical_indicators(ticker)
        if tech is not None:
            tech["ticker"] = ticker
            indicator_frames.append(tech)

    if indicator_frames:
        all_tech = pd.concat(indicator_frames, ignore_index=True)
        all_tech["Date"] = pd.to_datetime(all_tech["Date"])

        # Merge on entry_date (the actual trading day we enter the position)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df = df.merge(
            all_tech,
            left_on=["ticker", "entry_date"],
            right_on=["ticker", "Date"],
            how="left",
        ).drop(columns=["Date"])

    # ── 3. Drop rows with missing technical indicators ─────────────────────
    before = len(df)
    df = df.dropna(subset=["RSI", "MACD"])
    df = df[df["Gunning_Fog"] != 0]
    print(f"\nDropped {before - len(df):,} rows with missing indicators")
    print(f"Final dataset: {len(df):,} rows")

    # ── 4. Summary ─────────────────────────────────────────────────────────
    print("\nFeature summary:")
    feat_cols = ["Gunning_Fog", "Flesch_Ease", "Sentiment",
                 "Diff_Word_Ratio", "RSI", "MACD", "Volume_Change"]
    print(df[feat_cols].describe().round(3).to_string())

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}  ({len(df):,} rows)")
    print("=" * 70)
    print("DONE — next step: 05_baseline_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()