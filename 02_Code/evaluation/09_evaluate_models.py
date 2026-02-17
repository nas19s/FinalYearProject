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


def load_finbert_val_predictions(device):
    """
    Reconstructs the same val split used in training, runs FinBERT inference,
    and returns (true_labels, hard_preds, prob_of_up).
    """
    print("Loading FinBERT model and tensors...")
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading FinBERT: {e}")
        return None, None, None

    if not os.path.exists(TENSOR_PATH):
        print(f"Tensor file not found: {TENSOR_PATH}")
        return None, None, None

    data       = torch.load(TENSOR_PATH, map_location="cpu")
    input_ids  = data["input_ids"]
    attn_mask  = data["attention_mask"]
    labels     = data["labels"]

    # Replicate the exact val split from 08_train_finbert.py
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

    all_preds, all_probs, all_labels = [], [], []

    print("Running FinBERT inference on validation set...")
    for batch in dataloader:
        ids, mask, lbls = [t.to(device) for t in batch]
        with torch.no_grad():
            logits = model(ids, token_type_ids=None, attention_mask=mask).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_probs.extend(probs[:, 1])
        all_preds.extend(preds)
        all_labels.extend(lbls.cpu().numpy())

    return all_labels, all_preds, all_probs


def get_baseline_predictions(val_size: int):
    """
    Trains Logistic Regression on the same feature set and returns predictions
    on a held-out test set of the same size as the FinBERT val set.
    """
    print("Running Baseline (Logistic Regression)...")

    df       = pd.read_csv(CSV_PATH)
    df_clean = df[df[TARGET_COL] != -1].copy()

    features = [
        "Gunning_Fog", "Flesch_Ease", "Sentiment",
        "Diff_Word_Ratio", "Word_Count", "RSI", "MACD", "Volume_Change",
    ]
    X = df_clean[features].fillna(0)
    y = df_clean[TARGET_COL]

    # Use same fraction as FinBERT val split for a fair comparison
    test_frac = val_size / len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=max(test_frac, 0.15),
        stratify=y,
        random_state=RANDOM_SEED,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return y_test.tolist(), preds.tolist(), probs.tolist()


def plot_confusion_matrices(y_bert, base_preds, bert_preds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Down", "Up"]

    for ax, preds, title, cmap in zip(
        axes,
        [base_preds, bert_preds],
        ["Baseline (Logistic Regression)", "FinBERT (Fine-tuned)"],
        ["Blues", "Greens"],
    ):
        cm = confusion_matrix(y_bert, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=labels, yticklabels=labels,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")


def plot_roc_curves(y_true, base_probs, bert_probs):
    plt.figure(figsize=(8, 6))

    for probs, label, ls, color in [
        (base_probs, "Baseline", "--", "steelblue"),
        (bert_probs, "FinBERT",  "-",  "green"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, linestyle=ls, color=color,
                 linewidth=2, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: FinBERT vs Baseline")
    plt.legend(loc="lower right")

    path = os.path.join(RESULTS_DIR, "roc_curve_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")


def main():
    device = get_device()
    print(f"Device: {device}\n")

    # ── FinBERT evaluation ────────────────────────────────────────────────────
    y_bert, bert_preds, bert_probs = load_finbert_val_predictions(device)
    if y_bert is None:
        print("Evaluation failed — check paths.")
        return

    print("\n── FinBERT Metrics ─────────────────────────────────────────────")
    print(classification_report(y_bert, bert_preds, target_names=["Down", "Up"], digits=3))

    # ── Baseline evaluation ───────────────────────────────────────────────────
    y_base, base_preds, base_probs = get_baseline_predictions(val_size=len(y_bert))

    print("\n── Baseline Metrics (Logistic Regression) ──────────────────────")
    print(classification_report(y_base, base_preds, target_names=["Down", "Up"], digits=3))

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Note: ROC uses FinBERT's y_true for both curves (same label distribution)
    plot_confusion_matrices(y_bert, base_preds[:len(y_bert)], bert_preds)
    plot_roc_curves(y_bert, base_probs[:len(y_bert)], bert_probs)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    from sklearn.metrics import f1_score, accuracy_score
    summary = pd.DataFrame([
        {
            "Model":    "Baseline (LR)",
            "Accuracy": accuracy_score(y_base, base_preds),
            "F1_macro": f1_score(y_base, base_preds, average="macro"),
            "F1_weighted": f1_score(y_base, base_preds, average="weighted"),
        },
        {
            "Model":    "FinBERT",
            "Accuracy": accuracy_score(y_bert, bert_preds),
            "F1_macro": f1_score(y_bert, bert_preds, average="macro"),
            "F1_weighted": f1_score(y_bert, bert_preds, average="weighted"),
        },
    ])
    csv_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved: {csv_path}")
    print(summary.to_string(index=False))
    print(f"\nAll results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()