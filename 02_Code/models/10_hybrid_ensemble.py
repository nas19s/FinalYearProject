import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")
TENSOR_PATH  = os.path.join(PROJECT_ROOT, "03_Models", "finbert_tensors.pt")
CSV_PATH     = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "04_Results", "evaluation")

TARGET_COL  = "Label_Month"
BATCH_SIZE  = 8
VAL_SPLIT   = 0.15
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_finbert_probs(device):
    """
    Returns FinBERT's P(Up) for every sample in the val split,
    along with the val indices so we can align with the CSV features.
    """
    print("Loading FinBERT...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    data      = torch.load(TENSOR_PATH, map_location="cpu")
    input_ids = data["input_ids"]
    attn_mask = data["attention_mask"]
    labels    = data["labels"]

    # Replicate exact split from training
    indices = list(range(len(labels)))
    _, val_idx = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels.numpy(),
    )

    val_idx_t  = torch.tensor(val_idx)
    val_ds     = TensorDataset(input_ids[val_idx_t], attn_mask[val_idx_t], labels[val_idx_t])
    dataloader = DataLoader(val_ds, sampler=SequentialSampler(val_ds), batch_size=BATCH_SIZE)

    all_probs, all_labels = [], []

    print("Extracting FinBERT probability scores...")
    for batch in dataloader:
        ids, mask, lbls = [t.to(device) for t in batch]
        with torch.no_grad():
            logits = model(ids, token_type_ids=None, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs[:, 1])
        all_labels.extend(lbls.cpu().numpy())

    return val_idx, all_probs, all_labels


def main():
    print("=" * 60)
    print("Hybrid Ensemble: FinBERT + Technical Indicators")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}\n")

    # ── Step 1: Get FinBERT scores for val samples ────────────────────────────
    val_idx, finbert_probs, true_labels = extract_finbert_probs(device)

    # ── Step 2: Align CSV features with val indices ───────────────────────────
    df       = pd.read_csv(CSV_PATH)
    df_clean = df[df[TARGET_COL] != -1].copy().reset_index(drop=True)

    # The tensor file was built from df_clean in the same row order
    # val_idx references rows in df_clean
    val_df = df_clean.iloc[val_idx].copy().reset_index(drop=True)

    tech_features = ["RSI", "MACD", "Volume_Change", "Gunning_Fog", "Sentiment"]
    X_tech = val_df[tech_features].fillna(0).reset_index(drop=True)

    # ── Step 3: Build hybrid feature matrix ──────────────────────────────────
    X_hybrid = X_tech.copy()
    X_hybrid["FinBERT_Score"] = finbert_probs
    y = pd.Series(true_labels)

    print(f"Hybrid feature matrix: {X_hybrid.shape}")
    print(f"Label distribution:\n{y.value_counts()}\n")

    # ── Step 4: Train/test split of val set for meta-classifier ──────────────
    # We split the val set 50/50: half to fit the RF, half to evaluate it.
    # This is a standard stacking approach.
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        X_hybrid, y, test_size=0.5, random_state=RANDOM_SEED, stratify=y
    )

    # ── Step 5: Train Random Forest meta-classifier ───────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    rf.fit(X_meta_train, y_meta_train)
    hybrid_preds = rf.predict(X_meta_test)

    # ── Step 6: Baselines for comparison ─────────────────────────────────────
    # FinBERT alone (threshold 0.5)
    finbert_only = (X_meta_test["FinBERT_Score"] > 0.5).astype(int)

    # Technical indicators alone (no FinBERT score)
    X_tech_only = X_meta_test[tech_features]
    lr_tech = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_tech.fit(X_meta_train[tech_features], y_meta_train)
    tech_only_preds = lr_tech.predict(X_tech_only)

    # ── Step 7: Print results ─────────────────────────────────────────────────
    results = {
        "FinBERT alone":              finbert_only,
        "Technicals alone (LR)":      tech_only_preds,
        "Hybrid (FinBERT + Tech, RF)": hybrid_preds,
    }

    summary_rows = []
    for name, preds in results.items():
        print(f"\n── {name} ──────────────────────────────────────")
        print(classification_report(y_meta_test, preds, target_names=["Down", "Up"], digits=3))
        summary_rows.append({
            "Model": name,
            "F1_macro":    f1_score(y_meta_test, preds, average="macro"),
            "F1_weighted": f1_score(y_meta_test, preds, average="weighted"),
        })

    # ── Step 8: Feature importance plot ───────────────────────────────────────
    importance_df = pd.DataFrame({
        "Feature":    X_hybrid.columns,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n── Feature Importance ───────────────────────────────────────")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="steelblue")
    plt.xlabel("Importance")
    plt.title("Hybrid Model — Random Forest Feature Importance")
    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, "hybrid_feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    print(f"\nFeature importance plot saved: {fi_path}")

    # ── Step 9: Save summary ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    csv_path   = os.path.join(RESULTS_DIR, "hybrid_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()