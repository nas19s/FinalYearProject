import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import gc

# Project Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "03_Models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Config
HORIZON = "T20"
TARGET_COL = f"Label_{HORIZON}"
CHAMPION_DIR = os.path.join(MODELS_DIR, f"finbert_champion_{HORIZON}")
BATCH_SIZE = 16
DEVICE = torch.device("cpu")
torch.set_num_threads(8)

def main():
    print(f"Starting FinBERT Inference for {HORIZON}...")

    # Load test data
    test_path = os.path.join(DATA_DIR, "test.parquet")
    test_df = pd.read_parquet(test_path)
    
    # Filter for binary classification (remove FLAT labels)
    test_df = test_df[test_df[TARGET_COL] != 0].copy()
    test_df["binary_label"] = test_df[TARGET_COL].map({-1: 0, 1: 1})
    test_df = test_df.dropna(subset=["binary_label", "text"]).reset_index(drop=True)
    
    print(f"Loaded {len(test_df)} chunks across {test_df.groupby(['ticker','filing_date']).ngroups} filings.")

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(CHAMPION_DIR)
    all_input_ids, all_attention_masks = [], []
    texts = test_df["text"].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing"):
        batch = texts[i: i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        all_input_ids.append(enc["input_ids"])
        all_attention_masks.append(enc["attention_mask"])
        gc.collect()

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_masks = torch.cat(all_attention_masks, dim=0)
    labels = torch.tensor(test_df["binary_label"].astype(int).tolist(), dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = BertForSequenceClassification.from_pretrained(CHAMPION_DIR, num_labels=2)
    model.to(DEVICE)
    model.eval()

    # Inference Loop
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for input_ids_b, attn_b, labels_b in tqdm(loader, desc="Inference"):
            input_ids_b = input_ids_b.to(DEVICE)
            attn_b = attn_b.to(DEVICE)

            outputs = model(input_ids=input_ids_b, attention_mask=attn_b)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_b.numpy())

            del input_ids_b, attn_b, outputs
            gc.collect()

    # Save results with metadata
    results_df = test_df[["ticker", "filing_date", "section", "chunk_idx", "binary_label"]].copy()
    results_df["true_label"] = all_labels
    results_df["prob_up"] = all_probs
    results_df["pred_label"] = all_preds

    out_path = os.path.join(RESULTS_DIR, f"finbert_full_test_predictions_{HORIZON}.csv")
    results_df.to_csv(out_path, index=False)

    # Aggregate to filing level
    filing_level = (results_df
                    .groupby(["ticker", "filing_date"])
                    .agg(
                        mean_prob_up = ("prob_up", "mean"),
                        max_prob_up  = ("prob_up", "max"),
                        min_prob_up  = ("prob_up", "min"),
                        std_prob_up  = ("prob_up", "std"),
                        n_chunks     = ("prob_up", "count"),
                        true_label   = ("true_label", "first"),
                    )
                    .reset_index())

    filing_out = os.path.join(RESULTS_DIR, f"finbert_filing_level_probs_{HORIZON}.csv")
    filing_level.to_csv(filing_out, index=False)

    # Metric summary
    filing_level["filing_pred"] = (filing_level["mean_prob_up"] >= 0.5).astype(int)
    filing_acc = (filing_level["filing_pred"] == filing_level["true_label"]).mean()
    
    print(f"Filing-level accuracy (mean): {filing_acc:.4f}")
    print(f"Outputs saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()