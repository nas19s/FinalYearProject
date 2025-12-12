import pandas as pd
import os
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
METADATA_PATH = os.path.join(DATA_DIR, "sec_metadata.csv")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_PATH = os.path.join(DATA_DIR, "labeled_dataset.csv")

DRIFT_THRESHOLD = 0.02     # 1-day
WEEK_THRESHOLD = 0.03      # 5-day (optional)
MEDIUM_THRESHOLD = 0.06    # 20-day

def load_price_data(ticker):
    csv_path = os.path.join(PRICES_DIR, f"{ticker}_prices.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Date')

        df['Price_T1'] = df['Close'].shift(-1)
        df['Return_Short'] = (df['Price_T1'] - df['Close']) / df['Close']

        df['Price_T5'] = df['Close'].shift(-5)
        df['Return_Week'] = (df['Price_T5'] - df['Close']) / df['Close']

        df['Price_T20'] = df['Close'].shift(-20)
        df['Return_Month'] = (df['Price_T20'] - df['Close']) / df['Close']

        df = df.dropna(subset=['Return_Short', 'Return_Week', 'Return_Month'])

        return df[['Date', 'Close', 'Return_Short', 'Return_Week', 'Return_Month']]

    except Exception:
        return None

def categorize(val, threshold):
    if val > threshold:
        return 1
    elif val < -threshold:
        return 0
    else:
        return -1

def main():
    if not os.path.exists(METADATA_PATH):
        return

    meta_df = pd.read_csv(METADATA_PATH)
    meta_df['filing_date'] = pd.to_datetime(
        meta_df['filing_date'], format='%Y%m%d', errors='coerce'
    )
    meta_df = meta_df.dropna(subset=['filing_date'])
    meta_df['join_date'] = meta_df['filing_date'].dt.strftime('%Y-%m-%d')

    merged_data = []
    for ticker in tqdm(meta_df['ticker'].unique()):
        price_df = load_price_data(ticker)
        if price_df is None:
            continue

        filings = meta_df[meta_df['ticker'] == ticker]
        merged = pd.merge(
            filings,
            price_df,
            left_on='join_date',
            right_on='Date',
            how='inner'
        )

        if not merged.empty:
            merged_data.append(merged)

    if not merged_data:
        return

    final_df = pd.concat(merged_data, ignore_index=True)

    final_df['Label_Short'] = final_df['Return_Short'].apply(
        lambda x: categorize(x, DRIFT_THRESHOLD)
    )

    final_df['Label_Week'] = final_df['Return_Week'].apply(
        lambda x: categorize(x, WEEK_THRESHOLD)
    )

    final_df['Label_Month'] = final_df['Return_Month'].apply(
        lambda x: categorize(x, MEDIUM_THRESHOLD)
    )

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Samples:", len(final_df))
    print("Short:", final_df['Label_Short'].value_counts())
    print("Week:", final_df['Label_Week'].value_counts())
    print("Month:", final_df['Label_Month'].value_counts())

if __name__ == "__main__":
    main()
