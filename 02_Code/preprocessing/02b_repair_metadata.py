import os
import pandas as pd
import requests
import re
import glob
from tqdm import tqdm
import time
import json

# --- CONFIGURATION ---
# ⚠️ UPDATE THIS WITH YOUR EMAIL (Required by SEC)
USER_EMAIL = "nxa250@student.bham.ac.uk" 
USER_AGENT = f"Nasrudin Adan ({USER_EMAIL})"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "01_Data", "processed_sec")
METADATA_FILE = os.path.join(PROJECT_ROOT, "01_Data", "sec_metadata.csv")

HEADERS = {"User-Agent": USER_AGENT}

def get_cik_map():
    """Fetches the official Ticker -> CIK mapping from the SEC."""
    print("🌍 Fetching Ticker-CIK map from SEC.gov...")
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=HEADERS)
        data = resp.json()
        
        # Convert dictionary to Ticker -> CIK map
        cik_map = {}
        for entry in data.values():
            cik_map[entry['ticker']] = str(entry['cik_str']) # Keep as string, no leading zeros yet
        return cik_map
    except Exception as e:
        print(f"❌ Error fetching CIK map: {e}")
        return {}

def fetch_date_from_sec(cik, accession_number):
    """
    Constructs the SEC Archives URL and fetches ONLY the header 
    to find the 'FILED AS OF DATE'.
    """
    # SEC URL Structure requires Accession Number WITHOUT dashes
    accession_no_dashes = accession_number.replace("-", "")
    
    # URL to the raw text filing (we only read the first 5KB)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession_number}.txt"
    
    try:
        # Stream=True allows us to download only the beginning of the file
        with requests.get(url, headers=HEADERS, stream=True, timeout=10) as r:
            if r.status_code != 200:
                return None
            
            # Read first 2KB (Header is always at the top)
            chunk = next(r.iter_content(chunk_size=2048)).decode('utf-8', errors='ignore')
            
            # Regex to find date
            match = re.search(r"FILED AS OF DATE:\s+(\d{8})", chunk)
            if match:
                return match.group(1) # Returns YYYYMMDD
            
            # Fallback
            match = re.search(r"CONFORMED PERIOD OF REPORT:\s+(\d{8})", chunk)
            if match:
                return match.group(1)
                
    except Exception:
        pass
    return None

def repair_metadata():
    # 1. Load CIK Map
    cik_map = get_cik_map()
    if not cik_map:
        print("Stopping: Could not get CIK map.")
        return

    # 2. Load Existing Metadata
    if os.path.exists(METADATA_FILE):
        df_meta = pd.read_csv(METADATA_FILE)
        existing_filenames = set(df_meta['filename'].tolist())
        metadata_list = df_meta.to_dict('records')
    else:
        existing_filenames = set()
        metadata_list = []

    # 3. Scan for Orphan Files (Files in folder but NOT in CSV)
    print(f"🔍 Scanning {PROCESSED_DIR}...")
    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.txt"))
    
    orphans = []
    for f_path in all_files:
        filename = os.path.basename(f_path)
        if filename not in existing_filenames:
            orphans.append(filename)

    print(f"⚠️ Found {len(orphans)} orphan files (missing dates). Starting repair...")
    
    if len(orphans) == 0:
        print("✅ No repairs needed. Your metadata is complete.")
        return

    # 4. Repair Loop
    success_count = 0
    
    for filename in tqdm(orphans, desc="Fetching Dates"):
        try:
            # Parse filename: Ticker_Type_Accession.txt
            # Example: AAPL_10-K_0000320193-23-000106.txt
            name_parts = filename.replace('.txt', '').split('_')
            
            if len(name_parts) < 3:
                continue # Skip weird files

            ticker = name_parts[0]
            report_type = name_parts[1]
            filing_id = name_parts[2] # Accession Number

            # Get CIK
            if ticker not in cik_map:
                # Try handling cases like BRK-B vs BRK.B
                ticker_fix = ticker.replace("-", ".")
                if ticker_fix in cik_map:
                    cik = cik_map[ticker_fix]
                else:
                    continue # Can't find CIK, skip
            else:
                cik = cik_map[ticker]

            # Fetch Date from Web
            date_str = fetch_date_from_sec(cik, filing_id)
            
            if date_str:
                metadata_list.append({
                    "ticker": ticker,
                    "report_type": report_type,
                    "filing_date": date_str,
                    "filename": filename,
                    "filing_id": filing_id
                })
                success_count += 1
            
            # SEC Limit: Avoid hitting 10 req/sec. Small sleep is polite.
            time.sleep(0.15)
            
            # Save every 50 repairs to be safe
            if success_count % 50 == 0:
                pd.DataFrame(metadata_list).to_csv(METADATA_FILE, index=False)

        except Exception as e:
            # print(f"Error on {filename}: {e}")
            pass

    # 5. Final Save
    pd.DataFrame(metadata_list).to_csv(METADATA_FILE, index=False)
    print(f"\n✅ Repair Complete. Recovered dates for {success_count} files.")
    print(f"   Metadata saved to: {METADATA_FILE}")

if __name__ == "__main__":
    repair_metadata()