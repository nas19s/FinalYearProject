import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_finbert_predictions(device):
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading FinBERT: {e}")
        return None, None, None

    test_path = os.path.join(TENSOR_DIR, "test.pt")
    if not os.path.exists(test_path):
        return None, None, None
        
    data = torch.load(test_path)
    dataset = TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE)

    preds = []
    probs = []
    true_labels = []

    print("Running FinBERT inference...")
    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
            
        # Get probabilities (Class 1) and hard predictions
        batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        probs.extend(batch_probs[:, 1])
        preds.extend(batch_preds)
        true_labels.extend(b_labels.cpu().numpy())

    return true_labels, preds, probs

def get_baseline_predictions():
    print("Running Baseline (Logistic Regression)...")
    
    df = pd.read_csv(CSV_PATH)
    # Option A: Remove Neutrals
    df_clean = df[df[TARGET_COL] != -1].copy()
    
    features = ['Gunning_Fog', 'Flesch_Ease', 'Sentiment', 
                'Diff_Word_Ratio', 'Word_Count', 'RSI', 'MACD', 'Volume_Change']
    
    X = df_clean[features].fillna(0)
    y = df_clean[TARGET_COL]

    # Replicate exact split state used in FinBERT prep (random_state=42)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    _, X_test_final, _, y_test_final = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_full, y_train_full)
    
    preds = model.predict(X_test_final)
    probs = model.predict_proba(X_test_final)[:, 1]
    
    return y_test_final, preds, probs

def plot_confusion_matrices(y_true, base_preds, bert_preds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline
    cm_base = confusion_matrix(y_true, base_preds)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Baseline Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    # FinBERT
    cm_bert = confusion_matrix(y_true, bert_preds)
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("FinBERT Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_comparison.png"))

def plot_roc_curves(y_true, base_probs, bert_probs):
    plt.figure(figsize=(8, 6))
    
    # Baseline
    fpr_b, tpr_b, _ = roc_curve(y_true, base_probs)
    auc_b = auc(fpr_b, tpr_b)
    plt.plot(fpr_b, tpr_b, linestyle='--', label=f'Baseline (AUC = {auc_b:.2f})')
    
    # FinBERT
    fpr_f, tpr_f, _ = roc_curve(y_true, bert_probs)
    auc_f = auc(fpr_f, tpr_f)
    plt.plot(fpr_f, tpr_f, linewidth=2, color='green', label=f'FinBERT (AUC = {auc_f:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: FinBERT vs Baseline')
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_comparison.png"))

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    y_true_bert, bert_preds, bert_probs = load_finbert_predictions(device)
    y_true_base, base_preds, base_probs = get_baseline_predictions()
    
    if y_true_bert is None:
        print("Evaluation failed. Check data paths.")
        return

    print("\n--- BASELINE METRICS ---")
    print(classification_report(y_true_base, base_preds))
    
    print("\n--- FINBERT METRICS ---")
    print(classification_report(y_true_bert, bert_preds))

    plot_confusion_matrices(y_true_bert, base_preds, bert_preds)
    plot_roc_curves(y_true_bert, base_probs, bert_probs)

    print(f"\nEvaluation Complete. Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()