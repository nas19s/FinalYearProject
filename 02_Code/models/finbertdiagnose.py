"""
debug_merge.py  —  diagnose why labeled_dataset has 0 rows
"""
import pandas as pd
import os

PROJECT_ROOT = "/Users/nasrudinadan/Documents/FinalYearProject"
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")

# Load both sources
meta   = pd.read_csv(os.path.join(DATA_DIR, "sec_metadata.csv"))
chunks = pd.read_parquet(os.path.join(DATA_DIR, "sec_cleaned.parquet"))

print("=== METADATA ===")
print(f"Rows: {len(meta)}")
print(f"filing_date dtype : {meta['filing_date'].dtype}")
print(f"filing_date sample: {meta['filing_date'].head(3).tolist()}")
print(f"accession dtype   : {meta['accession'].dtype}")
print(f"accession sample  : {meta['accession'].head(3).tolist()}")

print("\n=== CHUNKS ===")
print(f"Rows: {len(chunks)}")
print(f"filing_date dtype : {chunks['filing_date'].dtype}")
print(f"filing_date sample: {chunks['filing_date'].head(3).tolist()}")
print(f"accession dtype   : {chunks['accession'].dtype}")
print(f"accession sample  : {chunks['accession'].head(3).tolist()}")

print("\n=== OVERLAP CHECK ===")
# Check if any ticker+date combos match between the two
meta_keys   = set(zip(meta["ticker"], meta["filing_date"].astype(str)))
chunk_keys  = set(zip(chunks["ticker"], chunks["filing_date"].astype(str)))
overlap     = meta_keys & chunk_keys
print(f"Unique (ticker, filing_date) in metadata : {len(meta_keys)}")
print(f"Unique (ticker, filing_date) in chunks   : {len(chunk_keys)}")
print(f"Overlapping keys                         : {len(overlap)}")
print(f"\nSample metadata keys : {list(meta_keys)[:3]}")
print(f"Sample chunk keys    : {list(chunk_keys)[:3]}")