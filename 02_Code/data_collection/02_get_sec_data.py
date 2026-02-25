import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from edgar import set_identity, Company

# --- Configuration ---
USER_AGENT_NAME = "Nasrudin Adan"
USER_AGENT_EMAIL = "nxa250@student.bham.ac.uk"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
SECTIONS_DIR = os.path.join(DATA_DIR, "sec_sections")
METADATA_FILE = os.path.join(DATA_DIR, "sec_metadata.csv")

FORM_LIMITS = {"10-K": 10, "10-Q": 40}

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
    "XOM", "JPM", "PG", "V", "LLY", "HD", "MA", "CVX", "MRK", "ABBV",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "WMT", "CRM",
    "BAC", "LIN", "PFE", "NFLX", "ADBE", "AMD", "DIS", "NKE", "ABT", "DHR",
    "TXN", "VZ", "NEE", "PM", "CMCSA", "UPS", "BMY", "RTX", "INTC", "HON",
]

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, "sec_download.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# Map internal keys to possible edgartools attribute names
SECTION_MAP = {
    "item_1a_risk_factors": ["risk_factors", "item1a"],
    "item_7_mda":           ["mda",           "item7"],
    "item_7a_market_risk":  ["market_risk",   "item7a"],
    "item_9a_controls":     ["controls",      "item9a"],
}

def _try_get(obj, attr_candidates):
    for attr in attr_candidates:
        try:
            val = getattr(obj, attr, None)
            if callable(val):
                val = val()
            if val and isinstance(val, str) and len(val.strip()) > 50:
                return val.strip()
        except Exception:
            pass
    return ""

def extract_sections(filing_obj):
    return {key: _try_get(filing_obj, attrs) for key, attrs in SECTION_MAP.items()}

def process_ticker(ticker, metadata_rows):
    try:
        company = Company(ticker)
    except Exception as e:
        log.warning(f"[{ticker}] Company init failed: {e}")
        return

    for form_type, limit in FORM_LIMITS.items():
        try:
            filings = company.get_filings(form=form_type).latest(limit)
        except Exception as e:
            log.warning(f"[{ticker}] get_filings({form_type}) failed: {e}")
            continue
            
        if not filings:
            continue

        filing_list = list(filings) if hasattr(filings, "__iter__") else [filings]
        
        for filing in filing_list:
            try:
                # Use filing_date (SEC receipt date) to avoid look-ahead bias
                raw_date = getattr(filing, "filing_date", None) or getattr(filing, "date", None)
                if raw_date is None:
                    continue

                filing_date = pd.to_datetime(str(raw_date)).strftime("%Y%m%d")
                accession = str(getattr(filing, "accession_number", "unknown")).replace("-", "")

                filing_obj = filing.obj()
                if filing_obj is None:
                    continue

                sections = extract_sections(filing_obj)
                total_chars = sum(len(v) for v in sections.values())
                
                if total_chars < 200:
                    continue

                fname = f"{ticker}_{form_type}_{filing_date}_{accession[:8]}.json"
                out_path = os.path.join(SECTIONS_DIR, fname)
                
                output_data = {
                    "ticker": ticker, 
                    "form_type": form_type,
                    "filing_date": filing_date, 
                    "accession": accession,
                    "sections": sections
                }

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                metadata_rows.append({
                    "ticker": ticker,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "accession": accession,
                    "section_file": fname,
                    "chars_risk": len(sections["item_1a_risk_factors"]),
                    "chars_mda": len(sections["item_7_mda"]),
                    "chars_market_risk": len(sections["item_7a_market_risk"]),
                    "chars_controls": len(sections["item_9a_controls"]),
                })
                
                log.info(f"[{ticker}] {form_type} {filing_date} - {total_chars:,} chars")

            except Exception as e:
                log.warning(f"[{ticker}] Filing error: {e}")

            time.sleep(0.35)

def main():
    set_identity(f"{USER_AGENT_NAME} {USER_AGENT_EMAIL}")
    os.makedirs(SECTIONS_DIR, exist_ok=True)

    metadata_rows = []
    for ticker in tqdm(TICKERS, desc="Processing Tickers"):
        process_ticker(ticker, metadata_rows)

    if metadata_rows:
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_csv(METADATA_FILE, index=False)
        print(f"\nMetadata saved to {METADATA_FILE} ({len(meta_df)} filings)")
        
        cols = ["chars_risk", "chars_mda", "chars_market_risk", "chars_controls"]
        for col in cols:
            pct = (meta_df[col] > 0).mean()
            print(f"{col}: {pct:.1%} coverage")
    else:
        print("\nNo data processed.")

if __name__ == "__main__":
    main()