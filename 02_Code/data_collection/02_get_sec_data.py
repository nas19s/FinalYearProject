import os
from sec_edgar_downloader import Downloader
from tqdm import tqdm

USER_AGENT_NAME = "Nasrudin Adan"
USER_AGENT_EMAIL = "nxa250@student.bham.ac.uk"  


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))


DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
SEC_DIR = os.path.join(DATA_DIR, "raw_sec")

# S&P Top 50 tickers
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
    "XOM", "JPM", "PG", "V", "LLY", "HD", "MA", "CVX", "MRK", "ABBV",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "WMT", "CRM",
    "BAC", "LIN", "PFE", "NFLX", "ADBE", "AMD", "DIS", "NKE", "ABT", "DHR",
    "TXN", "VZ", "NEE", "PM", "CMCSA", "UPS", "BMY", "RTX", "INTC", "HON"
]

def download_filings(ticker_list):
    print(f"Starting SEC filing download for {len(ticker_list)} companies...")
    print(f"Saving files to: {SEC_DIR}")
    print(f"Using User-Agent: {USER_AGENT_NAME} ({USER_AGENT_EMAIL})\n")
    
    
    dl = Downloader(USER_AGENT_NAME, USER_AGENT_EMAIL, SEC_DIR)
    
    for ticker in tqdm(ticker_list, desc="Downloading filings"):
        try:
            # Download last 10 years of annual reports (10-K)
            dl.get("10-K", ticker, limit=10)
            
            # Download last 10 years of quarterly reports (10-Q)
            dl.get("10-Q", ticker, limit=40)
            
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    print(f"\nSEC filing download complete. Check folder: {SEC_DIR}")

if __name__ == "__main__":
    # Safety check to prevent placeholder email usage
    if "CHANGE_ME" in USER_AGENT_EMAIL:
        print("STOP: You must update USER_AGENT_EMAIL with a real email address.")
        print("The SEC API will reject requests without a valid contact.")
    else:
        download_filings(tickers)
