import pandas as pd
import os
import textstat
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
TEXT_DIR = os.path.join(DATA_DIR, "processed_sec")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
INPUT_FILE = os.path.join(DATA_DIR, "labeled_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "final_feature_dataset.csv")


BUFFER_SIZE = 150000 
ANALYSIS_LIMIT = 100000 # NLP is performed on 100k chars

analyzer = SentimentIntensityAnalyzer()

def get_fast_nlp(filename):
    """
    Super-fast NLP extraction. Reads a buffer, cleans it, and scores it.
    """
    file_path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(file_path):
        return [0] * 5

    try:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            raw_chunk = f.read(BUFFER_SIZE)

        # 1. Quick Clean: Remove SEC Header & Tags
        # Find start of real text 
        start_match = re.search(r'<(DOCUMENT|TEXT|HTML)>', raw_chunk, re.IGNORECASE)
        prose = raw_chunk[start_match.start():] if start_match else raw_chunk
        
        # Strip HTML tags and collapse whitespace
        prose = re.sub(r'<[^>]+>', ' ', prose)
        prose = " ".join(prose.split())
        
        # Crop to the analysis limit
        prose = prose[:ANALYSIS_LIMIT]

        if len(prose) < 500:
            return [0] * 5

        # 2. Metrics
        fog = textstat.gunning_fog(prose)
        flesch = textstat.flesch_reading_ease(prose)
        
        # 3. Sentiment
        vs = analyzer.polarity_scores(prose)
        sentiment = vs['compound']
        
        # 4. Complexity
        diff_words = textstat.difficult_words(prose) / (len(prose.split()) + 1)
        word_count = len(prose.split())

        return [fog, flesch, sentiment, diff_words, word_count]

    except Exception:
        return [0] * 5

def get_technical_indicators(ticker):
    price_path = os.path.join(PRICES_DIR, f"{ticker}_prices.csv")
    if not os.path.exists(price_path): return None
    try:
        df = pd.read_csv(price_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('Date')
        
        # Tech Indicators
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        df['MACD'] = MACD(close=df['Close']).macd()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        return df[['Date', 'RSI', 'MACD', 'Volume_Change']]
    except Exception:
        return None

def main():
    print("🚀 Starting High-Speed Feature Engineering...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found!")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Faster NLP Loop
    tqdm.pandas(desc="Cleaning & NLP")
    nlp_results = df['filename'].progress_apply(get_fast_nlp)
    
    # Expand results into columns
    nlp_cols = ['Gunning_Fog', 'Flesch_Ease', 'Sentiment', 'Diff_Word_Ratio', 'Word_Count']
    df[nlp_cols] = pd.DataFrame(nlp_results.tolist(), index=df.index)
    
    # 2. Faster Market Loop
    unique_tickers = df['ticker'].unique()
    indicator_frames = []
    for ticker in tqdm(unique_tickers, desc="Technical Indicators"):
        tech_df = get_technical_indicators(ticker)
        if tech_df is not None:
            tech_df['ticker'] = ticker
            indicator_frames.append(tech_df)
            
    all_indicators = pd.concat(indicator_frames)
    
    # 3. Merge
    df['join_date'] = pd.to_datetime(df['join_date']).dt.strftime('%Y-%m-%d')
    final_df = pd.merge(df, all_indicators, left_on=['ticker', 'join_date'], right_on=['ticker', 'Date'], how='left')
    
    # 4. Final Cleanup
    final_df = final_df.dropna(subset=['RSI', 'Gunning_Fog'])
    final_df = final_df[final_df['Gunning_Fog'] != 0]

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Complete! Saved {len(final_df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()