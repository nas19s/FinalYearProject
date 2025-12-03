import pandas as pd
import os
from tqdm import tqdm

# Define file paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
METADATA_PATH = os.path.join(DATA_DIR, "sec_metadata.csv")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_PATH = os.path.join(DATA_DIR, "labeled_dataset.csv")

# Threshold for stock drift (1%)
DRIFT_THRESHOLD = 0.01

def load_price_data(ticker):
    """
    Reads the price CSV for a specific ticker.
    Returns a DataFrame with the Date, Close price, and calculated T+1 return.
    """
    csv_path = os.path.join(PRICES_DIR, f"{ticker}_prices.csv")
    
    if not os.path.exists(csv_path):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        
        # Standardize date format
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Date')
        
        # Calculate the return for the *next* day (T+1)
        # We shift the 'Close' column up by one to get tomorrow's price
        df['Next_Day_Price'] = df['Close'].shift(-1)
        df['T1_Return'] = (df['Next_Day_Price'] - df['Close']) / df['Close']
        
        # Drop the last row since it won't have a 'Next_Day_Price'
        df = df.dropna(subset=['T1_Return'])
        
        return df[['Date', 'Close', 'T1_Return']]
        
    except Exception as e:
        print(f"Error processing price data for {ticker}: {e}")
        return None

def main():
    print("Starting target definition process...")

    # 1. Load the metadata mapping file
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Could not find metadata file at {METADATA_PATH}")
        print("Please ensure you have run the metadata repair script first.")
        return

    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Basic validation
    if metadata_df.empty:
        print("Error: Metadata CSV is empty.")
        return

    # Ensure dates are in the correct format for merging
    # Errors='coerce' will turn bad dates into NaT (Not a Time), which we then drop
    metadata_df['filing_date'] = pd.to_datetime(metadata_df['filing_date'], format='%Y%m%d', errors='coerce')
    metadata_df = metadata_df.dropna(subset=['filing_date'])
    
    # Create a string column for the merge (YYYY-MM-DD)
    metadata_df['join_date'] = metadata_df['filing_date'].dt.strftime('%Y-%m-%d')

    print(f"Loaded {len(metadata_df)} filing records from metadata.")

    # 2. Merge filings with price data
    merged_data = []
    unique_tickers = metadata_df['ticker'].unique()
    
    print(f"Processing {len(unique_tickers)} tickers...")

    for ticker in tqdm(unique_tickers, desc="Merging data"):
        # Get the price history for this stock
        price_df = load_price_data(ticker)
        
        if price_df is None:
            continue
            
        # Get the filings corresponding to this stock
        stock_filings = metadata_df[metadata_df['ticker'] == ticker].copy()
        
        # Left join: Keep the filing info, attach price info where dates match
        # We use inner join here to ensure we only keep filings where we have price data
        merged = pd.merge(stock_filings, price_df, left_on='join_date', right_on='Date', how='inner')
        
        if not merged.empty:
            merged_data.append(merged)

    # 3. Compile and save results
    if not merged_data:
        print("Warning: No matches found between filings and price data.")
        print("Check if your price CSVs and metadata dates cover the same time periods.")
        return

    final_df = pd.concat(merged_data, ignore_index=True)

    # Create the classification target
    # 1 = Price went UP > 1%
    # 0 = Price went DOWN < -1%
    # -1 = Neutral (small movement)
    def categorize_drift(return_val):
        if return_val > DRIFT_THRESHOLD:
            return 1
        elif return_val < -DRIFT_THRESHOLD:
            return 0
        else:
            return -1

    final_df['Target_Label'] = final_df['T1_Return'].apply(categorize_drift)

    # Save to disk
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 30)
    print("Processing complete.")
    print(f"Total labeled samples created: {len(final_df)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("-" * 30)
    
    # Show distribution of targets
    print("Class distribution:")
    print(final_df['Target_Label'].value_counts())

if __name__ == "__main__":
    main()