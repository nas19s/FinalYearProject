import os
import time
import datetime
import gc

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
TENSOR_PATH    = os.path.join(PROJECT_ROOT, "03_Models", "finbert_tensors.pt")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE        = 2
GRAD_ACCUMULATION = 4      # effective batch = 8
EPOCHS            = 8      
PATIENCE          = 2
LEARNING_RATE     = 2e-5
VAL_SPLIT         = 0.15
RANDOM_SEED       = 42
WEIGHT_STRATEGY   = 'squared'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def format_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple Metal (MPS) acceleration.")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")


def load_tensors():
    if not os.path.exists(TENSOR_PATH):
        raise FileNotFoundError(f"Tensor file not found: {TENSOR_PATH}")

    print(f"Loading tensors from: {TENSOR_PATH}")
    data = torch.load(TENSOR_PATH, map_location="cpu")

    input_ids      = data["input_ids"]
    attention_mask = data["attention_mask"]
    labels         = data["labels"]

    n = len(labels)
    print(f"Total samples: {n}")
    unique, counts = torch.unique(labels, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"  Label {u}: {c} samples ({c/n*100:.1f}%)")

    indices = list(range(n))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels.numpy(),
    )

    def make_dataset(idx_list):
        idx = torch.tensor(idx_list)
        return TensorDataset(input_ids[idx], attention_mask[idx], labels[idx])

    train_ds = make_dataset(train_idx)
    val_ds   = make_dataset(val_idx)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    return train_ds, val_ds, labels


def compute_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    unique, counts = torch.unique(labels, return_counts=True)
    n_total   = float(len(labels))
    n_classes = len(unique)

    raw_weights = n_total / (n_classes * counts.float())

    if WEIGHT_STRATEGY == 'sqrt':
        raw_weights = torch.sqrt(raw_weights)
    elif WEIGHT_STRATEGY == 'squared':
        raw_weights = raw_weights ** 2

    raw_weights = raw_weights / raw_weights.mean()

    ordered = torch.zeros(n_classes)
    for cls, w in zip(unique.tolist(), raw_weights.tolist()):
        ordered[cls] = w

    print(f"Class weights ({WEIGHT_STRATEGY}): Down={ordered[0]:.3f}  Up={ordered[1]:.3f}")
    return ordered.to(device)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        ids, mask, lbls = [b.to(device) for b in batch]
        with torch.no_grad():
            out = model(ids, token_type_ids=None, attention_mask=mask)
        preds = torch.argmax(out.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    f1_wtd   = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return acc, f1_wtd, f1_macro, all_labels, all_preds


def train():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    train_ds, val_ds, all_labels = load_tensors()

    train_loader = DataLoader(
        train_ds, sampler=RandomSampler(train_ds), batch_size=BATCH_SIZE
    )
    val_loader = DataLoader(
        val_ds, sampler=SequentialSampler(val_ds), batch_size=BATCH_SIZE
    )

    # ── Resume from checkpoint if available, else start fresh ─────────────────
    checkpoint = os.path.join(MODEL_SAVE_DIR, "config.json")
    if os.path.exists(checkpoint):
        print(f"\nResuming from saved checkpoint: {MODEL_SAVE_DIR}")
        print("(Epochs 1-6 already complete — running epochs 7-8 only)")
        model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    else:
        print("\nNo checkpoint found — starting from ProsusAI/finbert base.")
        model = BertForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
    model.to(device)

    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps  = (len(train_loader) // GRAD_ACCUMULATION) * EPOCHS
    warmup_steps = total_steps // 10
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    class_weights = compute_class_weights(all_labels, device)
    loss_fct      = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Start best_val_f1_macro from last run's result so we only save improvements
    best_val_f1_macro = 0.5968
    epochs_no_improve = 0
    start_time        = time.time()

    print(f"\nStarting training — up to {EPOCHS} epochs, effective batch {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"Previous best macro F1: {best_val_f1_macro} (from epoch 6)")
    print(f"Early stopping patience: {PATIENCE}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1} / {EPOCHS}")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids, mask, lbls = [b.to(device) for b in batch]

            out  = model(ids, token_type_ids=None, attention_mask=mask)
            loss = loss_fct(out.logits, lbls) / GRAD_ACCUMULATION

            total_loss += loss.item()
            loss.backward()

            if (step + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        avg_loss = total_loss / len(train_loader) * GRAD_ACCUMULATION
        elapsed  = format_time(time.time() - start_time)
        print(f"  Avg train loss : {avg_loss:.4f}  |  Time: {elapsed}")

        val_acc, val_f1_wtd, val_f1_macro, true_labels, pred_labels = evaluate(
            model, val_loader, device
        )
        print(f"  Val accuracy   : {val_acc:.4f}")
        print(f"  Val F1 weighted: {val_f1_wtd:.4f}")
        print(f"  Val F1 macro   : {val_f1_macro:.4f}  <- tracked for early stopping")
        print(classification_report(
            true_labels, pred_labels,
            target_names=["Down", "Up"],
            digits=3,
        ))

        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            epochs_no_improve = 0
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"  ✓ New best model saved (macro F1 = {best_val_f1_macro:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    print("\n" + "=" * 60)
    print(f"Training complete.  Total time: {format_time(time.time() - start_time)}")
    print(f"Best Val Macro F1 : {best_val_f1_macro:.4f}")
    print(f"Model saved to    : {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    train()