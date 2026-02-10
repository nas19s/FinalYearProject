import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
PERIOD = "10y"

os.makedirs(PRICES_DIR, exist_ok=True)

# S&P 50 tickers (representative large-cap stocks)
SP50_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
    "XOM", "JPM", "PG", "V", "LLY", "HD", "MA", "CVX", "MRK", "ABBV",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "WMT", "CRM",
    "BAC", "LIN", "PFE", "NFLX", "ADBE", "AMD", "DIS", "NKE", "ABT", "DHR",
    "TXN", "VZ", "NEE", "PM", "CMCSA", "UPS", "BMY", "RTX", "INTC", "HON"
]

def get_price_data(tickers):
    """Download historical price data for a list of tickers and save to CSVs."""
    print(f"Downloading {PERIOD} of price data for {len(tickers)} stocks...")
    print(f"Saving files to: {PRICES_DIR}\n")
    
    # Fetch all tickers at once for speed
    all_data = yf.download(tickers, period=PERIOD, group_by='ticker', auto_adjust=True)
    
    saved_count = 0
    for ticker in tqdm(tickers, desc="Saving CSVs"):
        try:
            df = all_data[ticker]
            if not df.empty:
                file_path = os.path.join(PRICES_DIR, f"{ticker}_prices.csv")
                df.to_csv(file_path)
                saved_count += 1
        except KeyError:
            print(f"Warning: No data for {ticker}")
    
    print(f"\nFinished. Saved {saved_count}/{len(tickers)} tickers to {PRICES_DIR}")

if __name__ == "__main__":
    get_price_data(SP50_TICKERS)
