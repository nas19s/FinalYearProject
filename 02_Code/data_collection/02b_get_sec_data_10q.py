import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from edgar import set_identity, Company

# SEC Identity setup
set_identity("Nasrudin Adan nxa250@student.bham.ac.uk")

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
SECTIONS_DIR = os.path.join(DATA_DIR, "sec_sections")
METADATA_FILE = os.path.join(DATA_DIR, "sec_metadata.csv")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
    "XOM", "JPM", "PG", "V", "LLY", "HD", "MA", "CVX", "MRK", "ABBV",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "WMT", "CRM",
    "BAC", "LIN", "PFE", "NFLX", "ADBE", "AMD", "DIS", "NKE", "ABT", "DHR",
    "TXN", "VZ", "NEE", "PM", "CMCSA", "UPS", "BMY", "RTX", "INTC", "HON",
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, "sec_download_10q.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

# Extraction mapping
DICT_KEYS = {
    "item_1a_risk_factors": ["Item 1A", "Item 1A.", "item 1a"],
    "item_7_mda":           ["Item 7",  "Item 7.",  "mda"],
    "item_7a_market_risk":  ["Item 7A", "Item 7A.", "item 7a"],
    "item_9a_controls":     ["Item 9A", "Item 9A.", "item 9a"],
}

SECTION_KEYS = {
    "item_1a_risk_factors": "part_i_item_1a",
    "item_7_mda":           "part_ii_item_7",
    "item_7a_market_risk":  "part_ii_item_7a",
    "item_9a_controls":     "part_ii_item_9a",
}

PROPERTY_NAMES = {
    "item_1a_risk_factors": "risk_factors",
    "item_7_mda":           "management_discussion",
    "item_7a_market_risk":  None,
    "item_9a_controls":     None,
}

def _to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    for method in ("to_text", "to_markdown", "get_text"):
        try:
            result = getattr(val, method)()
            if isinstance(result, str) and len(result.strip()) > 50:
                return result.strip()
        except Exception:
            pass
    try:
        t = val.text
        if isinstance(t, str) and len(t.strip()) > 50:
            return t.strip()
    except Exception:
        pass
    return str(val).strip()

def extract_sections(filing_obj) -> dict:
    sections = {}
    for section_key in DICT_KEYS:
        text = ""
        for key in DICT_KEYS[section_key]:
            if text: break
            try:
                text = _to_str(filing_obj[key])
            except (KeyError, TypeError):
                pass
        if not text:
            try:
                text = _to_str(filing_obj.sections[SECTION_KEYS[section_key]])
            except (KeyError, TypeError, AttributeError):
                pass
        if not text:
            prop = PROPERTY_NAMES.get(section_key)
            if prop:
                try:
                    val = getattr(filing_obj, prop, None)
                    if callable(val): val = val()
                    text = _to_str(val)
                except Exception:
                    pass
        sections[section_key] = text
    return sections

def main():
    print("Starting 10-Q Section Download...")
    os.makedirs(SECTIONS_DIR, exist_ok=True)

    # Load existing metadata for appending
    existing_meta = pd.read_csv(METADATA_FILE)
    existing_files = set(existing_meta["section_file"].tolist())
    print(f"Found {len(existing_meta)} existing filings in metadata.")

    new_rows = []

    for ticker in tqdm(TICKERS, desc="Processing Tickers"):
        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-Q").latest(40)
        except Exception as e:
            log.warning(f"[{ticker}] 10-Q download failed: {e}")
            continue

        if not filings:
            continue

        filing_list = list(filings) if hasattr(filings, "__iter__") else [filings]

        for filing in filing_list:
            try:
                raw_date = (getattr(filing, "filing_date", None)
                            or getattr(filing, "date", None))
                if raw_date is None:
                    continue

                filing_date = pd.to_datetime(str(raw_date)).strftime("%Y%m%d")
                accession = str(getattr(filing, "accession_number", "unknown")).replace("-", "")

                fname = f"{ticker}_10-Q_{filing_date}_{accession[:8]}.json"

                if fname in existing_files:
                    continue

                filing_obj = filing.obj()
                if filing_obj is None:
                    continue

                sections = extract_sections(filing_obj)
                total_chars = sum(len(v) for v in sections.values())

                if total_chars < 100:
                    continue

                out_path = os.path.join(SECTIONS_DIR, fname)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "ticker": ticker, 
                            "form_type": "10-Q",
                            "filing_date": filing_date, 
                            "accession": accession,
                            "sections": sections
                        },
                        f, ensure_ascii=False, indent=2,
                    )

                new_rows.append({
                    "ticker": ticker,
                    "form_type": "10-Q",
                    "filing_date": filing_date,
                    "accession": accession,
                    "section_file": fname,
                    "chars_risk": len(sections["item_1a_risk_factors"]),
                    "chars_mda": len(sections["item_7_mda"]),
                    "chars_market_risk": len(sections["item_7a_market_risk"]),
                    "chars_controls": len(sections["item_9a_controls"]),
                })
                log.info(f"[{ticker}] 10-Q {filing_date} saved: {total_chars:,} chars")

            except Exception as e:
                log.warning(f"[{ticker}] Error processing filing: {e}")

            time.sleep(0.35)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing_meta, new_df], ignore_index=True)
        combined = combined.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
        combined.to_csv(METADATA_FILE, index=False)

        print(f"\nDownload finished.")
        print(f"Added {len(new_rows)} new 10-Q filings.")
        print(f"Total entries now: {len(combined)}")
        
        print("\nNew section coverage:")
        for col in ["chars_risk", "chars_mda", "chars_market_risk", "chars_controls"]:
            pct = (new_df[col] > 0).mean()
            print(f"  {col:20}: {pct:.1%}")
    else:
        print("\nNo new filings found to download.")

if __name__ == "__main__":
    main()