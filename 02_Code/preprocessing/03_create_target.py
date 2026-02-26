import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths and Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")

METADATA_FILE = os.path.join(DATA_DIR, "sec_metadata.csv")
CHUNKS_FILE = os.path.join(DATA_DIR, "sec_cleaned.parquet")
PRICE_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_FILE = os.path.join(DATA_DIR, "labeled_dataset.parquet")

HORIZONS = [5, 10, 20]
VOL_WINDOW = 20

# Split boundaries
VAL_START = "2021-01-01"
TEST_START = "2023-01-01"

def load_price_data(ticker):
    for suffix in ("_prices.csv", ".csv"):
        fpath = os.path.join(PRICE_DIR, f"{ticker}{suffix}")
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, parse_dates=["Date"])
                df = df.sort_values("Date").drop_duplicates(subset=["Date"])
                df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
                df["Rolling_Vol"] = df["Daily_Return"].rolling(VOL_WINDOW).std()
                return df
            except Exception as e:
                print(f"[{ticker}] Error loading prices: {e}")
                return None
    return None

def is_after_hours(ts):
    if pd.isnull(ts):
        return False
    try:
        # SEC filings after 4PM ET are treated as next-day impact
        return ts.hour >= 16
    except AttributeError:
        return False

def get_entry(price_df, anchor_date, advance_one=False):
    mask = price_df["Date"] > anchor_date if advance_one else price_df["Date"] >= anchor_date
    valid = price_df[mask]
    if valid.empty:
        return None, None
    
    row = valid.iloc[0]
    entry_price = row["Open"] if "Open" in price_df.columns else row["Close"]
    return row["Date"], float(entry_price)

def get_forward_close(price_df, entry_date, horizon):
    future = price_df[price_df["Date"] > entry_date]
    if len(future) < horizon:
        return np.nan
    return float(future.iloc[horizon - 1]["Close"])

def classify(fwd_return, rolling_vol):
    if pd.isna(fwd_return) or pd.isna(rolling_vol) or rolling_vol == 0:
        return np.nan
    if fwd_return > rolling_vol:
        return 1
    if fwd_return < -rolling_vol:
        return -1
    return 0

def main():
    # 1. Load data
    meta = pd.read_csv(METADATA_FILE)
    meta["filing_date"] = pd.to_datetime(
        meta["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    meta = meta.dropna(subset=["filing_date"]).sort_values("filing_date")

    chunks = pd.read_parquet(CHUNKS_FILE)
    chunks["filing_date"] = pd.to_datetime(
        chunks["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # 2. Process tickers
    label_rows = []
    tickers = meta["ticker"].unique()

    for ticker in tqdm(tickers, desc="Processing Tickers"):
        price_df = load_price_data(ticker)
        if price_df is None:
            continue

        ticker_filings = meta[meta["ticker"] == ticker]

        for _, filing in ticker_filings.iterrows():
            f_date = filing["filing_date"]
            after_hrs = is_after_hours(f_date)

            entry_date, entry_price = get_entry(price_df, f_date, advance_one=after_hrs)
            if entry_date is None or not entry_price:
                continue

            vol_row = price_df[price_df["Date"] == entry_date]
            rolling_vol = float(vol_row["Rolling_Vol"].values[0]) if not vol_row.empty else np.nan

            row = {
                "ticker": ticker,
                "form_type": filing["form_type"],
                "filing_date": f_date,
                "accession": filing["accession"],
                "section_file": filing["section_file"],
                "entry_date": entry_date,
                "entry_price": entry_price,
                "after_hours": after_hrs,
            }

            for h in HORIZONS:
                fwd_close = get_forward_close(price_df, entry_date, h)
                fwd_return = (
                    (fwd_close - entry_price) / entry_price
                    if not np.isnan(fwd_close) else np.nan
                )
                row[f"fwd_return_T{h}"] = fwd_return
                row[f"Label_T{h}"] = classify(fwd_return, rolling_vol)

            label_rows.append(row)

    labels_df = pd.DataFrame(label_rows)
    label_cols = [f"Label_T{h}" for h in HORIZONS]
    labels_df = labels_df.dropna(subset=label_cols)

    # 3. Merge chunks
    labels_df["filing_date_str"] = labels_df["filing_date"].dt.strftime("%Y%m%d")
    chunks["filing_date_str"] = (chunks["filing_date"].astype(str)
                                  .str.strip()
                                  .str.replace("-", "", regex=False))

    labeled = labels_df.merge(
        chunks[[
            "ticker", "filing_date_str",
            "section", "section_weight",
            "chunk_idx", "n_chunks", "text", "token_len"
        ]],
        on=["ticker", "filing_date_str"],
        how="inner",
    )

    labeled["filing_date"] = pd.to_datetime(labeled["filing_date_str"], format="%Y%m%d")
    labeled = labeled.drop(columns=["filing_date_str"])

    # 4. Time-based split
    train = labeled[labeled["filing_date"] < VAL_START]
    val = labeled[(labeled["filing_date"] >= VAL_START) &
                  (labeled["filing_date"] < TEST_START)]
    test = labeled[labeled["filing_date"] >= TEST_START]

    # 5. Output
    labeled.to_parquet(OUTPUT_FILE, index=False)
    train.to_parquet(os.path.join(DATA_DIR, "train.parquet"), index=False)
    val.to_parquet(os.path.join(DATA_DIR, "val.parquet"), index=False)
    test.to_parquet(os.path.join(DATA_DIR, "test.parquet"), index=False)

    print(f"Finished. Total chunks: {len(labeled)}")

if __name__ == "__main__":
    main()