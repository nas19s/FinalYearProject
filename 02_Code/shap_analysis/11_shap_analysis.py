import shap
import torch
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
TEXT_DIR     = os.path.join(DATA_DIR, "processed_sec")
CSV_PATH     = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "04_Results", "shap")

SAMPLE_SIZE       = 5
TARGET_COL        = 'Label_Month'
WORDS_WANTED      = 350
TOC_SKIP_FRACTION = 0.05
MIN_WORDS         = 120   # raised from 60 — short extracts miss financial content

os.makedirs(RESULTS_DIR, exist_ok=True)

# Financial signal words — if a passage lacks these, it's description not results
FINANCIAL_SIGNALS = re.compile(
    r'\b(revenue|revenues|net\s+sales|income|earnings|loss|losses|margin|margins|'
    r'growth|decline|declin|decreas|increas|operating|cash\s+flow|quarter|fiscal|'
    r'year.over.year|compared\s+to|period|results?\s+of\s+operations|'
    r'basis\s+points|guidance|outlook|headwind|tailwind|impairment|charge)\b',
    re.IGNORECASE,
)

BOILERPLATE_RE = re.compile(
    r'forward[\s\-]+looking\s+statements?'
    r'|private\s+securities\s+litigation\s+reform\s+act'
    r'|safe\s+harbor'
    r'|table\s+of\s+contents'
    r'|this\s+(annual|quarterly)\s+report\s+on\s+form\s+10'
    r'|incorporated\s+herein\s+by\s+reference'
    r'|refers\s+collectively\s+to'
    r'|fiscal\s+(year|calendar)',
    re.IGNORECASE,
)
TOC_LINE_RE = re.compile(r'\bItem\b.{5,120}?\s+\d{1,3}\s*$', re.IGNORECASE)
XBRL_RE     = re.compile(r'(us-gaap:|dei:|[a-z]{2,6}:[A-Z][a-zA-Z]{5,}|0000\d{6}\s)')


def is_garbage(text: str) -> bool:
    if len(text) < 10:
        return True
    non_alpha = sum(1 for c in text if not (c.isalpha() or c.isspace() or c in '.,;:()%-'))
    return (non_alpha / len(text)) > 0.25


def normalise(raw: str) -> str:
    text = re.sub(r'<[^>]{0,200}>', ' ', raw)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&[a-zA-Z]{2,8};', ' ', text)
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    return ' '.join(text.split())


def find_prose_start(text: str) -> int:
    for marker in [r'<TEXT>', r'<DOCUMENT>', r'Item\s+1\.\s+Business']:
        matches = list(re.finditer(marker, text, re.IGNORECASE))
        if matches:
            return matches[0].start()
    for start in range(0, min(len(text), 500_000), 5000):
        chunk = text[start: start + 5000]
        if XBRL_RE.search(chunk):
            continue
        if sum(c.isalpha() or c.isspace() for c in chunk) / len(chunk) > 0.60:
            return start
    return 0


def is_toc_region(window: str) -> bool:
    lines = re.split(r'\s{2,}|\n', window.strip())
    if len(lines) < 3:
        return bool(re.search(r'\s\d{1,3}\s*$', window.strip()))
    toc_hits = sum(1 for l in lines if re.search(r'\s\d{1,3}\s*$', l.strip()))
    return (toc_hits / len(lines)) > 0.35


def find_real_section(text: str, pattern: re.Pattern) -> int:
    min_start = int(len(text) * TOC_SKIP_FRACTION)
    for match in pattern.finditer(text):
        if match.start() < min_start:
            continue
        body_start = match.end()
        window     = text[body_start: body_start + 800]
        if is_toc_region(window):
            continue
        if XBRL_RE.search(window):
            continue
        if is_garbage(window):
            continue
        if sum(c.isalpha() for c in window) / max(len(window), 1) < 0.40:
            continue
        return body_start
    return -1


def find_financial_paragraph(text: str, start: int, char_budget: int = 15000) -> int:
    """
    Starting from `start`, scan forward in paragraph-sized chunks.
    Return the position of the first chunk that contains >= 2 financial
    signal words. This skips the business-description opener that many
    MD&A sections begin with before getting to actual financial results.
    """
    chunk_size = 500   # ~80 words per chunk
    end        = min(start + char_budget, len(text))

    pos = start
    while pos < end:
        chunk = text[pos: pos + chunk_size]
        if len(FINANCIAL_SIGNALS.findall(chunk)) >= 2:
            return pos
        # Advance by half a chunk so we don't miss a paragraph boundary
        pos += chunk_size // 2

    # Nothing found — return original start (better than nothing)
    return start


def clean_snippet(raw: str, words_wanted: int = WORDS_WANTED) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    good = [
        s for s in sentences
        if not is_garbage(s)
        and not BOILERPLATE_RE.search(s)
        and not TOC_LINE_RE.search(s)
        and not XBRL_RE.search(s)
        and len(s.split()) > 3
    ]
    return ' '.join(' '.join(good).split()[:words_wanted])


def extract_section(filename: str) -> str:
    path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARN] File not found: {filename}")
        return ""

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    text        = normalise(raw)
    prose_start = find_prose_start(text)
    text        = text[prose_start:]

    candidates = [
        ("MD&A (Item 7)",
         re.compile(r"\bITEM\s+7\.?\s+MANAGEMENT", re.IGNORECASE)),
        ("MD&A 10-Q (Item 2)",
         re.compile(r"\bITEM\s+2\.?\s+MANAGEMENT", re.IGNORECASE)),
        ("Risk Factors (Item 1A)",
         re.compile(r"\bITEM\s+1A\.?\s+RISK\s+FACTOR", re.IGNORECASE)),
    ]

    for section_name, pattern in candidates:
        section_start = find_real_section(text, pattern)
        if section_start == -1:
            continue

        # Skip past any business-description opener to find financial results
        financial_start = find_financial_paragraph(text, section_start)
        if financial_start != section_start:
            print(f"  [SKIP opener] Advanced {financial_start - section_start} chars to financial content")

        result = clean_snippet(text[financial_start: financial_start + WORDS_WANTED * 7])

        if len(result.split()) < MIN_WORDS:
            print(f"  [SKIP] '{section_name}' in {filename} — only {len(result.split())} words.")
            continue

        sig_count = len(FINANCIAL_SIGNALS.findall(result))
        print(f"  [OK]  '{section_name}' from {filename} "
              f"({len(result.split())} words, {sig_count} financial signals)")
        return result

    # Fallback
    print(f"  [FALLBACK] {filename} — using 20% skip.")
    words  = text.split()
    skip   = int(len(words) * 0.20) if len(words) > 5000 else min(200, len(words) // 4)
    result = clean_snippet(' '.join(words[skip: skip + WORDS_WANTED * 7]))
    return result if len(result.split()) > 30 else ""


def main():
    print("=" * 60)
    print("SHAP Analysis — FinBERT Token Explanations (v5)")
    print("=" * 60)

    device = torch.device("cpu")

    tok_path  = os.path.join(MODEL_PATH, "vocab.txt")
    tokenizer = (
        BertTokenizer.from_pretrained(MODEL_PATH)
        if os.path.exists(tok_path)
        else BertTokenizer.from_pretrained("ProsusAI/finbert")
    )

    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("Model loaded.\n")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    def predict_pipe(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        inputs = tokenizer(
            [str(t) for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=1).detach().numpy()

    df      = pd.read_csv(CSV_PATH)
    down_df = df[df[TARGET_COL] == 0]
    samples = (
        down_df.sample(SAMPLE_SIZE, random_state=42)
        if len(down_df) >= SAMPLE_SIZE else down_df
    )

    print(f"Extracting text from {len(samples)} downside filings...\n")
    text_data, meta = [], []
    for _, row in samples.iterrows():
        snippet = extract_section(row['filename'])
        if len(snippet.split()) >= 30:
            text_data.append(snippet)
            meta.append(row['filename'])
        else:
            print(f"  [DROP] {row['filename']}\n")

    if not text_data:
        print("[ERROR] No usable text extracted.")
        return

    print("\n── Snippet previews (first 400 chars) ──────────────────────────")
    for fname, t in zip(meta, text_data):
        sig_count = len(FINANCIAL_SIGNALS.findall(t))
        print(f"FILE: {fname}  [{sig_count} financial signals]")
        print(f"      {t[:400]}\n")

    print(f"Running SHAP on {len(text_data)} documents (~2 min each on CPU)...")
    masker      = shap.maskers.Text(tokenizer)
    explainer   = shap.Explainer(predict_pipe, masker)
    shap_values = explainer(text_data)

    html_path = os.path.join(RESULTS_DIR, "shap_explanation.html")
    with open(html_path, "w", encoding='utf-8') as f:
        f.write(shap.plots.text(shap_values[0], display=False))
    print(f"\nToken HTML  → {html_path}")

    bar_path = os.path.join(RESULTS_DIR, "shap_bar_summary.png")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values.mean(0), max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Bar summary → {bar_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()