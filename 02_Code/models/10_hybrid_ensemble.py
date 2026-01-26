import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
TENSOR_DIR = os.path.join(DATA_DIR, "finbert_tensors")
CSV_PATH = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "evaluation")

TARGET_COL = 'Label_Month'
BATCH_SIZE = 8

def get_finbert_probs(model, device, dataloader):
    model.eval()
    probs = []
    
    print("Extracting FinBERT signals...")
    for batch in dataloader:
        b_input_ids, b_input_mask, _ = [t.to(device) for t in batch]
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
            
        # Softmax to get probability of Class 1 (Up)
        batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs.extend(batch_probs[:, 1])
        
    return probs

def main():
    print("Training Hybrid Ensemble (Text + Math)...")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Reconstruct the Data Split to match training phase exactly
    df = pd.read_csv(CSV_PATH)
    # Option A: Remove Neutrals
    df_clean = df[df[TARGET_COL] != -1].copy()
    
    # Split 1: 80/20
    train_df, test_df = train_test_split(df_clean, test_size=0.2, stratify=df_clean[TARGET_COL], random_state=42)
    # Split 2: 50/50 split of the remaining 20%
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df[TARGET_COL], random_state=42)
    
    print(f"Targeting Test Set: {len(test_df)} samples")

    # Load FinBERT
    try:
        bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        bert_model.to(device)
    except Exception as e:
        print(f"Error loading FinBERT: {e}")
        return

    # Load Tensors
    test_pt_path = os.path.join(TENSOR_DIR, "test.pt")
    if not os.path.exists(test_pt_path):
        print("Error: Test tensors not found.")
        return
        
    data = torch.load(test_pt_path)
    
    if len(data['input_ids']) != len(test_df):
        print("Mismatch between tensors and dataframe. Aborting.")
        return
        
    dataset = TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE)

    # Get FinBERT probabilities
    finbert_probs = get_finbert_probs(bert_model, device, dataloader)
    
    # Prepare Hybrid Features (Math + AI Score)
    feature_cols = ['RSI', 'MACD', 'Volume_Change', 'Gunning_Fog']
    X_hybrid = test_df[feature_cols].copy()
    X_hybrid['FinBERT_Score'] = finbert_probs 
    
    y_test = test_df[TARGET_COL]

    # Split Test set to train the Meta-Classifier
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        X_hybrid, y_test, test_size=0.5, random_state=42
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_meta_train, y_meta_train)
    
    # Evaluate Hybrid Model
    y_pred = rf_model.predict(X_meta_test)
    
    print("\n--- Hybrid Model Results (FinBERT + RSI + MACD) ---")
    print(classification_report(y_meta_test, y_pred))
    
    # Evaluate FinBERT Alone (for comparison)
    finbert_preds_only = (X_meta_test['FinBERT_Score'] > 0.5).astype(int)
    print("\n--- FinBERT Alone (On this subset) ---")
    print(classification_report(y_meta_test, finbert_preds_only))

    # Feature Importance
    importance = pd.DataFrame({
        'Feature': X_hybrid.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    importance.to_csv(os.path.join(RESULTS_DIR, "hybrid_feature_importance.csv"))

if __name__ == "__main__":
    main()