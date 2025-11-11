import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm

# --- SETTINGS ---
DATA_DIR = "../../01_Data"
PRICES_DIR = os.path.join(DATA_DIR, "prices")
PERIOD = "10y"  # Fetch 10 years of historical data

# Make sure the directory exists
os.makedirs(PRICES_DIR, exist_ok=True)

# S&P 50 tickers – a representative sample of large, liquid companies
SP50_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
    "XOM", "JPM", "PG", "V", "LLY", "HD", "MA", "CVX", "MRK", "ABBV",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "WMT", "CRM",
    "BAC", "LIN", "PFE", "NFLX", "ADBE", "AMD", "DIS", "NKE", "ABT", "DHR",
    "TXN", "VZ", "NEE", "PM", "CMCSA", "UPS", "BMY", "RTX", "INTC", "HON"
]

def download_prices(tickers):
    """
    Grab historical price data for a list of tickers and save each to a CSV.
    """
    print(f"Fetching {PERIOD} of price data for {len(tickers)} stocks...")
    print("This includes major events like the 2020 crash and recent market swings.\n")

    # Download all tickers at once for speed
    all_data = yf.download(tickers, period=PERIOD, group_by='ticker', auto_adjust=True)
    
    saved_count = 0

    for ticker in tqdm(tickers, desc="Saving CSVs"):
        try:
            df = all_data[ticker]
            if not df.empty:
                csv_path = os.path.join(PRICES_DIR, f"{ticker}_prices.csv")
                df.to_csv(csv_path)
                saved_count += 1
        except KeyError:
            print(f"Warning: Could not find data for {ticker}")
    
    print(f"\nDone! Successfully saved {saved_count}/{len(tickers)} tickers to:")
    print(f"  {os.path.abspath(PRICES_DIR)}\n")

if __name__ == "__main__":
    print("Starting download process...\n")
    download_prices(SP50_TICKERS)
    
 