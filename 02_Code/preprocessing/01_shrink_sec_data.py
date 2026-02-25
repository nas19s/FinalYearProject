"""
01_shrink_sec_data.py  —  SEC text cleaning & chunking pipeline
"""

import os, re, json, logging
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
SECTIONS_DIR = os.path.join(DATA_DIR, "sec_sections")
METADATA_FILE = os.path.join(DATA_DIR, "sec_metadata.csv")   # FIX 1: drive from metadata
OUTPUT_FILE  = os.path.join(DATA_DIR, "sec_cleaned.parquet")

MODEL_NAME    = "ProsusAI/finbert"
MAX_TOKENS    = 512
STRIDE        = 128
MIN_CHUNK_LEN = 64

SECTION_WEIGHTS = {
    "item_1a_risk_factors": 1.0,
    "item_7_mda":           1.0,
    "item_7a_market_risk":  0.6,
    "item_9a_controls":     0.8,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_RE_HTML       = re.compile(r"<[^>]+>")
_RE_HTML_ENT   = re.compile(r"&[a-zA-Z0-9#]+;")
_RE_EXHIBIT    = re.compile(r"(exhibit|signature|pursuant to|incorporated by reference).*",
                             re.IGNORECASE)
_RE_NUM_LINE   = re.compile(r"^\s*[\d\s\$\%\,\.\-\(\)]+\s*$", re.MULTILINE)
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    text = _RE_HTML.sub(" ", raw)
    text = _RE_HTML_ENT.sub(" ", text)
    text = text.replace("\xa0", " ")          # FIX 2: remove non-breaking spaces
    text = text.replace("\u200b", "")         # zero-width spaces (also common in SEC filings)
    text = _RE_EXHIBIT.sub("", text)
    text = _RE_NUM_LINE.sub("", text)
    text = text.replace("$", " dollars ").replace("%", " percent ")
    text = _RE_WHITESPACE.sub(" ", text).strip()
    sentences = [s.strip() for s in text.split(".") if len(s.split()) >= 6]
    return ". ".join(sentences)


def stride_chunks(text: str, tokenizer):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids), MAX_TOKENS - STRIDE):
        window = token_ids[start: start + MAX_TOKENS]
        if len(window) < MIN_CHUNK_LEN:
            break
        decoded = tokenizer.decode(window, skip_special_tokens=True)
        chunks.append(decoded)
    return chunks if chunks else [text[:3000]]


def main():
    print("=" * 70)
    print("SEC TEXT CLEANING & CHUNKING PIPELINE")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # FIX 1: read only files that are in trimmed metadata (1,823 filings)
    meta = pd.read_csv(METADATA_FILE)
    valid_files = set(meta["section_file"].tolist())
    json_files  = [f for f in os.listdir(SECTIONS_DIR)
                   if f.endswith(".json") and f in valid_files]

    print(f"Processing {len(json_files)} filings (from metadata, pre-2016 excluded)")
    print(f"Skipping {len([f for f in os.listdir(SECTIONS_DIR) if f.endswith('.json')]) - len(json_files)} excluded files")

    rows = []

    for fname in tqdm(json_files, desc="Cleaning"):
        fpath = os.path.join(SECTIONS_DIR, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            log.warning(f"Could not read {fname}: {e}")
            continue

        ticker      = doc.get("ticker", "")
        form_type   = doc.get("form_type", "")
        filing_date = doc.get("filing_date", "")
        accession   = doc.get("accession", "")
        sections    = doc.get("sections", {})

        for section_key, section_weight in SECTION_WEIGHTS.items():
            raw_text = sections.get(section_key, "")
            if not raw_text:
                continue
            clean = clean_text(raw_text)
            if len(clean.split()) < 30:
                continue

            chunks = stride_chunks(clean, tokenizer)
            for chunk_idx, chunk_text in enumerate(chunks):
                rows.append({
                    "ticker":         ticker,
                    "form_type":      form_type,
                    "filing_date":    filing_date,
                    "accession":      accession,
                    "section":        section_key,
                    "section_weight": section_weight,
                    "chunk_idx":      chunk_idx,
                    "n_chunks":       len(chunks),
                    "text":           chunk_text,
                    "token_len":      len(tokenizer.tokenize(chunk_text)),
                })

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df):,} chunks from {df['filing_date'].nunique():,} unique filing dates")
    print(f"Output -> {OUTPUT_FILE}")
    print("\nToken length stats:")
    print(df["token_len"].describe().to_string())
    print("\nRows per section:")
    print(df.groupby("section")["ticker"].count().to_string())
    print("\nSample cleaned text (first chunk of first row):")
    print(repr(df["text"].iloc[0][:200]))


if __name__ == "__main__":
    main()