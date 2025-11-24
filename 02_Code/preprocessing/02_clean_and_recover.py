import os
import glob
import re
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

RAW_DIR = os.path.join(PROJECT_ROOT, "01_Data", "raw_sec")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "01_Data", "processed_sec")
METADATA_FILE = os.path.join(PROJECT_ROOT, "01_Data", "sec_metadata.csv")

# Skip files larger than 100MB to be safe
MAX_FILE_SIZE_MB = 100 

os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_filing_date(raw_content):
    """Finds date in the first 5000 characters."""
    # Look for FILED AS OF DATE
    match = re.search(r"FILED AS OF DATE:\s+(\d{8})", raw_content[:5000])
    if match:
        return match.group(1)
    
    # Fallback
    match = re.search(r"CONFORMED PERIOD OF REPORT:\s+(\d{8})", raw_content[:5000])
    if match:
        return match.group(1)
    return None

def clean_text_regex(text):
    """
    The 'Nuclear' option: Uses Regex to strip HTML tags.
    Much faster and crash-proof compared to BeautifulSoup.
    """
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_and_recover():
    # Load existing metadata
    if os.path.exists(METADATA_FILE):
        metadata_df = pd.read_csv(METADATA_FILE)
        metadata_list = metadata_df.to_dict('records')
    else:
        metadata_list = []

    files = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True)
    print(f"Found {len(files)} raw files. Switching to Regex cleaning...")

    for file_path in tqdm(files, desc="Processing"):
        try:
            # 1. Size Check
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                tqdm.write(f"⚠️ Skipping Large File ({file_size_mb:.2f} MB): {os.path.basename(file_path)}")
                os.remove(file_path)
                continue

            # 2. Extract Filename Info
            parts = file_path.split(os.sep)
            # Handle different path depths safely
            if len(parts) >= 4:
                ticker = parts[-4]
                report_type = parts[-3]
                filing_id = parts[-2]
            else:
                # Fallback for unexpected folder structure
                ticker = "UNKNOWN"
                report_type = "UNKNOWN"
                filing_id = os.path.basename(file_path).replace('.txt', '')

            # 3. Read Content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            # 4. Extract Date & Clean (The fast part)
            filing_date = extract_filing_date(raw_content)
            clean_text = clean_text_regex(raw_content)

            # 5. Save Processed File
            new_filename = f"{ticker}_{report_type}_{filing_id}.txt"
            save_path = os.path.join(PROCESSED_DIR, new_filename)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            # 6. Update Metadata
            metadata_list.append({
                "ticker": ticker,
                "report_type": report_type,
                "filing_date": filing_date,
                "filename": new_filename,
                "filing_id": filing_id
            })

            # 7. Delete Raw File
            os.remove(file_path)

            # Save metadata periodically
            if len(metadata_list) % 100 == 0:
                pd.DataFrame(metadata_list).to_csv(METADATA_FILE, index=False)

        except Exception as e:
            tqdm.write(f"❌ Error processing {os.path.basename(file_path)}: {e}")

    # Final Save
    pd.DataFrame(metadata_list).to_csv(METADATA_FILE, index=False)
    
    # Clean empty folders
    for root, dirs, _ in os.walk(RAW_DIR, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass

    print(f"\n✅ Success! Processed {len(metadata_list)} files.")
    print(f"Metadata saved to: {METADATA_FILE}")

if __name__ == "__main__":
    clean_and_recover()