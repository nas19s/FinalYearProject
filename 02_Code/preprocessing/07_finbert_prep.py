import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
TEXT_DIR     = os.path.join(DATA_DIR, "processed_sec")
CSV_PATH     = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
OUT_DIR      = os.path.join(PROJECT_ROOT, "03_Models")

TARGET_COL   = "Label_Month"
MAX_LEN      = 512
WORDS_WANTED = 350      # safely under 512 BERT tokens
VALIDATION_SAMPLE = 5  # files to inspect in the validation preview

os.makedirs(OUT_DIR, exist_ok=True)

# ── Boilerplate patterns ───────────────────────────────────────────────────────
BOILERPLATE_RE = re.compile(
    r'forward[\s\-]+looking\s+statements?'
    r'|private\s+securities\s+litigation\s+reform\s+act'
    r'|safe\s+harbor'
    r'|table\s+of\s+contents'
    r'|incorporated\s+herein\s+by\s+reference'
    r'|refers\s+collectively\s+to'
    r'|fiscal\s+(year|calendar)',
    re.IGNORECASE,
)

# XBRL taxonomy line: contains CIK-style numbers and us-gaap: prefixes
XBRL_LINE_RE = re.compile(
    r'(us-gaap:|dei:|[a-z]{2,6}:[A-Z][a-zA-Z]{5,}|0000\d{6}\s)',
)


# ── Core helpers ───────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    # Decode common HTML entities that survive BeautifulSoup
    text = re.sub(r'&#\d+;', ' ', text)       # numeric entities &#160; &#8217;
    text = re.sub(r'&[a-z]+;', ' ', text)     # named entities &amp; &nbsp;
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def find_prose_start(text: str) -> int:
   
    # Try explicit document boundary tags (kept by BeautifulSoup as text)
    for marker in [r'<TEXT>', r'<DOCUMENT>', r'Item\s+1\.\s+Business']:
        matches = list(re.finditer(marker, text, re.IGNORECASE))
        if matches:
            return matches[0].start()

    # Fallback: scan in 5,000-char windows until alphanum ratio > 60%
    window = 5000
    for start in range(0, min(len(text), 500_000), window):
        chunk = text[start: start + window]
        if XBRL_LINE_RE.search(chunk):
            continue  # still in XBRL block
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in chunk) / len(chunk)
        if alpha_ratio > 0.60:
            return start

    return 0  # give up, start from beginning


def is_toc_window(window: str) -> bool:
   
    # Split on double-spaces or newlines
    lines = re.split(r'\s{2,}|\n', window.strip())
    if len(lines) < 2:
        # Single-line — check if it ends in a short page number
        return bool(re.search(r'\s\d{1,3}\s*$', window.strip()))
    toc_hits = sum(1 for ln in lines if re.search(r'\s\d{1,3}\s*$', ln.strip()))
    return (toc_hits / len(lines)) > 0.30


def find_section_body(text: str, pattern: re.Pattern) -> int:
    
    for m in pattern.finditer(text):
        body_start = m.end()
        window     = text[body_start: body_start + 600]

        if is_toc_window(window):
            continue

        # Check for XBRL taxonomy noise
        if XBRL_LINE_RE.search(window):
            continue

        alpha = sum(c.isalpha() for c in window) / max(len(window), 1)
        if alpha < 0.40:
            continue

        return body_start

    return -1


def clean_body(raw: str, words: int = WORDS_WANTED) -> str:
   
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    good = [
        s for s in sentences
        if not BOILERPLATE_RE.search(s)
        and not XBRL_LINE_RE.search(s)
        and len(s.split()) > 3       # drop stub fragments
    ]
    combined = ' '.join(good)
    return ' '.join(combined.split()[:words])


def extract_section(filename: str) -> str:
    
    path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    text = normalise(raw)

    # ── Skip XBRL preamble ────────────────────────────────────────────────────
    prose_start = find_prose_start(text)
    text = text[prose_start:]

    # ── Section header patterns ───────────────────────────────────────────────
    # Note: files use UPPER CASE headers (ITEM 7.) — re.IGNORECASE handles both
    candidates = [
        ("MD&A 10-K (Item 7)",
         re.compile(r"\bITEM\s+7\.?\s+MANAGEMENT", re.IGNORECASE)),

        ("MD&A 10-Q (Item 2)",
         re.compile(r"\bITEM\s+2\.?\s+MANAGEMENT", re.IGNORECASE)),

        ("Risk Factors (Item 1A)",
         re.compile(r"\bITEM\s+1A\.?\s+RISK\s+FACTOR", re.IGNORECASE)),
    ]

    for section_name, pattern in candidates:
        start = find_section_body(text, pattern)
        if start == -1:
            continue

        raw_body = text[start: start + WORDS_WANTED * 7]
        result   = clean_body(raw_body)

        if len(result.split()) < 60:
            continue   # too short after cleaning, try next candidate

        return result

    # ── Fallback: proportional skip ───────────────────────────────────────────
    words  = text.split()
    total  = len(words)
    skip   = int(total * 0.20) if total > 5000 else min(200, total // 4)
    raw_fb = ' '.join(words[skip: skip + WORDS_WANTED * 7])
    result = clean_body(raw_fb)
    return result if len(result.split()) > 30 else ""


# ── Validation preview ────────────────────────────────────────────────────────

def run_validation(df: pd.DataFrame) -> float:
    """
    Inspect a sample of files and report extraction quality.
    Returns the fraction of sampled files with OK extraction.
    """
    sample = df.sample(min(VALIDATION_SAMPLE, len(df)), random_state=42)

    SECTIONS = {
        "Business":     re.compile(r'\bITEM\s+1\.?\s+BUSINESS\b', re.IGNORECASE),
        "Risk Factors": re.compile(r'\bITEM\s+1A\.?\s+RISK', re.IGNORECASE),
        "MD&A":         re.compile(r'\bITEM\s+(?:7|2)\.?\s+MANAGEMENT', re.IGNORECASE),
        "Financials":   re.compile(r'\bITEM\s+(?:8|1)\.?\s+FINANCIAL\s+STATEMENT', re.IGNORECASE),
    }

    bar = "─" * 60
    header = "\n" + "═"*70 + "\n  VALIDATION PREVIEW  —  Checking text extraction quality\n" + "═"*70
    print(header)

    ok_count = 0
    for _, row in sample.iterrows():
        fname = row["filename"]
        path  = os.path.join(TEXT_DIR, fname)
        print(f"\n{bar}")
        print(f"File   : {fname}")

        if not os.path.exists(path):
            print("  !! FILE NOT FOUND !!")
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        text = normalise(raw)
        prose_start = find_prose_start(text)
        prose = text[prose_start:]

        fmt = "plain"
        print(f"Format : {fmt}   |   Raw chars: {len(raw):,}   |   Prose chars (after XBRL skip): {len(prose):,}")
        print(f"XBRL preamble skipped: {prose_start:,} chars  ({prose_start/max(len(raw),1)*100:.1f}%)")

        wc = {}
        for sec_name, pat in SECTIONS.items():
            start = find_section_body(prose, pat)
            if start != -1:
                body  = clean_body(prose[start: start + WORDS_WANTED * 7])
                wc[sec_name] = len(body.split())
            else:
                wc[sec_name] = 0

        print("Section word counts:")
        for sec_name, count in wc.items():
            status = "✓" if count >= 100 else "✗ LOW"
            print(f"   {sec_name:15s}: {count:6,} words  {status}")

        combined = sum(wc.values())
        ok = combined >= 200
        if ok:
            ok_count += 1
        print(f"Combined words : {combined}   |   Extraction OK: {ok}")

        # Show a snippet of what we'd actually feed to FinBERT
        snippet = extract_section(fname)
        if snippet:
            print(f"\n  → FinBERT snippet preview (first 200 chars):")
            print(f"    {snippet[:200]}")
        else:
            print("\n  !! NO TEXT EXTRACTED — check file path and format !!")

    ok_rate = ok_count / max(len(sample), 1)
    print("\n" + "═"*70)
    print(f"  Validation: {ok_count}/{len(sample)} files OK ({ok_rate*100:.0f}%)")
    print("═"*70 + "\n")
    return ok_rate


# ── Full-dataset audit ────────────────────────────────────────────────────────

def run_audit(df: pd.DataFrame):
    print("\n" + "═"*70)
    print("  FULL-DATASET EXTRACTION AUDIT")
    print("═"*70)
    print("Running quick audit over all files (no tokenization yet)...")

    results  = {"ok": 0, "failed": 0, "missing": 0}
    failures = []

    for fname in tqdm(df["filename"], total=len(df)):
        path = os.path.join(TEXT_DIR, fname)
        if not os.path.exists(path):
            results["missing"] += 1
            failures.append(fname)
            continue

        snippet = extract_section(fname)
        if len(snippet.split()) >= 60:
            results["ok"] += 1
        else:
            results["failed"] += 1
            failures.append(fname)

    total = len(df)
    ok_pct = results["ok"] / max(total, 1) * 100
    print(f"\nExtraction OK    : {results['ok']} / {total} ({ok_pct:.1f}%)")
    print(f"Failed           : {results['failed']}")
    print(f"Missing files    : {results['missing']}")

    if failures[:10]:
        print(f"\nFirst failed files: {failures[:10]}")

    if ok_pct < 80:
        print(f"\n[!] WARNING: {100-ok_pct:.0f}% of files failed extraction.")
    else:
        print(f"\n[✓] Extraction quality acceptable ({ok_pct:.1f}% success).")

    return ok_pct


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_and_save(df: pd.DataFrame, tokenizer):
    print("\nStarting full tokenization...")

    texts  = []
    labels = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        snippet = extract_section(row["filename"])
        if len(snippet.split()) >= 30:
            texts.append(snippet)
            labels.append(int(row[TARGET_COL]))
            valid_indices.append(idx)

    print(f"Valid texts: {len(texts)} / {len(df)}")

    print("Tokenizing...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    save_path = os.path.join(OUT_DIR, "finbert_tensors.pt")
    torch.save({
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels":         labels_tensor,
        "valid_indices":  valid_indices,
    }, save_path)

    print(f"\n✓ Tensors saved to: {save_path}")
    print(f"  Shape: {encodings['input_ids'].shape}")
    print(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  07_finbert_prep.py  —  SEC Section Extraction + Tokenization")
    print("=" * 70 + "\n")

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df[TARGET_COL] != -1].copy()
    print(f"Loaded {len(df)} binary samples (neutrals removed).")
    print(f"Class distribution:\n{df[TARGET_COL].value_counts()}\n")

    # Step 1: Validation preview
    run_validation(df)

    # Step 2: Full audit
    ok_pct = run_audit(df)

    # Step 3: Tokenize (ask user if audit looks bad)
    if ok_pct < 50:
        ans = input(
            f"\n[!] Only {ok_pct:.0f}% of files extracted successfully.\n"
            "Proceed to full tokenization and tensor saving? [y/N]: "
        )
        if ans.strip().lower() != "y":
            print("Aborted.")
            return
    
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    tokenize_and_save(df, tokenizer)


if __name__ == "__main__":
    main()