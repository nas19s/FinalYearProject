import pandas as pd
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
TEXT_DIR = os.path.join(DATA_DIR, "processed_sec")
INPUT_FILE = os.path.join(DATA_DIR, "final_feature_dataset.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "finbert_tensors")

# Target Column (Must match what we used in Baseline)
TARGET_COL = 'Label_Month'

# BERT settings
MAX_LEN = 512  # FinBERT's maximum input length
MODEL_NAME = "ProsusAI/finbert"

def load_text(filename):
    """
    Reads the text file associated with a row.
    We only read the first 5000 characters to save time, 
    as BERT only sees the first 512 tokens anyway.
    """
    path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(5000) 
    except Exception:
        return ""

def run_prep():
    print("Starting FinBERT Data Preparation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Dataset
    if not os.path.exists(INPUT_FILE):
        print("Error: Input file not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for Option A (Remove Neutrals)
    df_clean = df[df[TARGET_COL] != -1].copy()
    print(f"Loaded {len(df_clean)} binary samples (Neutrals removed).")

    # 2. Split Data (Train / Val / Test)
    # We do this BEFORE tokenization to ensure no data leakage
    # 80% Train, 10% Validation, 10% Test
    train_df, test_df = train_test_split(df_clean, test_size=0.2, stratify=df_clean[TARGET_COL], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df[TARGET_COL], random_state=42)
    
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Calculate Class Weights (The Fix for Imbalance)
    # This tells us how much to penalize the model for missing the minority class
    y_train = train_df[TARGET_COL].values
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"Computed Class Weights: {weights_dict}")
    print("   (Class 0 'Down' weight is higher to force model to learn it)")

    # 4. Tokenization Loop
    print("Loading FinBERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def process_split(subset_df, split_name):
        input_ids = []
        attention_masks = []
        labels = []
        
        print(f"Tokenizing {split_name} set...")
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
            # Read text
            text = load_text(row['filename'])
            
            # Tokenize
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(row[TARGET_COL])

        # Stack into tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        
        # Save to disk
        torch.save({
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }, os.path.join(OUTPUT_DIR, f"{split_name}.pt"))

    # Process all splits
    process_split(train_df, "train")
    process_split(val_df, "val")
    process_split(test_df, "test")

    # Save weights so we can load them during training
    torch.save(torch.tensor(class_weights, dtype=torch.float), os.path.join(OUTPUT_DIR, "class_weights.pt"))

    print("-" * 30)
    print(f"Data Preparation Complete.")
    print(f"Tensors saved to: {OUTPUT_DIR}")
    print("Ready for FinBERT training.")

if __name__ == "__main__":
    run_prep()