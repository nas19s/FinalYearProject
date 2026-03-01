"""
07_finbert_prep.py  —  Tokenisation Pipeline

"""

import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "03_Models")

MODEL_NAME = "ProsusAI/finbert"
MAX_LEN    = 512
os.makedirs(MODELS_DIR, exist_ok=True)


def prep_split(df, tokenizer, split_name, target_col):
    df = df[df[target_col] != 0].copy()
    df["binary_label"] = df[target_col].map({-1: 0, 1: 1})
    df = df.dropna(subset=["binary_label", "text"])

    print(f"\n  [{split_name}] {len(df):,} chunks after dropping FLAT")
    print(f"  Label balance: DOWN={(df['binary_label']==0).sum():,}  "
          f"UP={(df['binary_label']==1).sum():,}")

    texts  = df["text"].tolist()
    labels = df["binary_label"].astype(int).tolist()

    all_input_ids, all_attention_mask = [], []
    batch_size = 64

    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"  Tokenising {split_name}", leave=False):
        batch = texts[i: i + batch_size]
        enc   = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])

    return {
        "input_ids":      torch.cat(all_input_ids,      dim=0),
        "attention_mask": torch.cat(all_attention_mask, dim=0),
        "labels":         torch.tensor(labels, dtype=torch.long),
        "ticker":         df["ticker"].tolist(),
        "filing_date":    df["filing_date"].astype(str).tolist(),
        "section":        df["section"].tolist(),
        "section_weight": df["section_weight"].tolist(),
        "chunk_idx":      df["chunk_idx"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="T5",
                        choices=["T5", "T10", "T20"],
                        help="Prediction horizon (T5, T10, or T20)")
    args = parser.parse_args()

    target_col = f"Label_{args.horizon}"

    print("=" * 70)
    print(f"07_finbert_prep.py  —  Horizon: {args.horizon}  Target: {target_col}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer loaded: {MODEL_NAME}")

    for split_name, fpath in [
        ("train", os.path.join(DATA_DIR, "train.parquet")),
        ("val",   os.path.join(DATA_DIR, "val.parquet")),
        ("test",  os.path.join(DATA_DIR, "test.parquet")),
    ]:
        print(f"\nProcessing {split_name}...")
        df = pd.read_parquet(fpath)
        print(f"  Loaded {len(df):,} chunks")

        tensors = prep_split(df, tokenizer, split_name, target_col)

        # Save with horizon suffix so all three coexist on disk
        out_path = os.path.join(MODELS_DIR,
                                f"finbert_tensors_{split_name}_{args.horizon}.pt")
        torch.save(tensors, out_path)
        print(f"  Saved -> {out_path}")
        print(f"  Tensor shape: {tensors['input_ids'].shape}")

    print("\n" + "=" * 70)
    print(f"DONE — now run: python 08_train_finbert.py --horizon {args.horizon}")
    print("=" * 70)


if __name__ == "__main__":
    main()