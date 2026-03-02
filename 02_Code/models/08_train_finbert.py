import os
import gc
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, roc_auc_score)
from tqdm import tqdm

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
MODELS_DIR = os.path.join(PROJECT_ROOT, '03_Models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, '04_Results', 'metrics')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
MODEL_NAME = 'ProsusAI/finbert'
EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUM = 1
LR = 2e-5
WARMUP_FRAC = 0.1
MAX_GRAD_NORM = 1.0
PATIENCE = 2
MAX_TRAIN_SAMPLES = 4000
MAX_VAL_SAMPLES = 1000
MAX_TEST_SAMPLES = 4000

# Global device config
DEVICE = torch.device('cpu')
torch.set_num_threads(8)

def free_memory():
    gc.collect()

def load_tensors(split, horizon, max_samples=None):
    path = os.path.join(MODELS_DIR, f'finbert_tensors_{split}_{horizon}.pt')
    data = torch.load(path, map_location="cpu")
    total = data["labels"].shape[0]

    if max_samples and max_samples < total:
        labels = data["labels"]
        down_idx = (labels == 0).nonzero(as_tuple=True)[0]
        up_idx = (labels == 1).nonzero(as_tuple=True)[0]

        n_each = max_samples // 2
        down_sampled = down_idx[torch.randperm(len(down_idx))[:n_each]]
        up_sampled = up_idx[torch.randperm(len(up_idx))[:n_each]]
        idx = torch.cat([down_sampled, up_sampled])
        idx = idx[torch.randperm(len(idx))]

        data = {
            k: v[idx] if isinstance(v, torch.Tensor) else
               [v[i] for i in idx.tolist()] if isinstance(v, list) else v
            for k, v in data.items()
        }
        print(f"Sampled {max_samples} from {total} (Stratified: {n_each} per class)")

    dataset = TensorDataset(
        data["input_ids"],
        data["attention_mask"],
        data["labels"],
    )
    return dataset, data["labels"]

def compute_class_weights(labels_tensor):
    counts = torch.bincount(labels_tensor)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(counts)
    return weights

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            del input_ids, attention_mask, labels, outputs
            free_memory()

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float("nan")

    return avg_loss, f1, auc, all_preds, all_labels, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="T5", choices=["T5", "T10", "T20"])
    args = parser.parse_args()
    horizon = args.horizon

    champion_dir = os.path.join(MODELS_DIR, f"finbert_champion_{horizon}")
    os.makedirs(champion_dir, exist_ok=True)

    print(f'Starting training for horizon: {horizon}')
    
    # Load data
    train_dataset, train_labels = load_tensors("train", horizon, max_samples=MAX_TRAIN_SAMPLES)
    val_dataset, _ = load_tensors("val", horizon, max_samples=MAX_VAL_SAMPLES)
    test_dataset, _ = load_tensors("test", horizon, max_samples=MAX_TEST_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Init model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # Layer freezing: keep only top layers and heads trainable
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(layer in name for layer in ['classifier', 'pooler', 'encoder.layer.11', 'encoder.layer.10']):
            param.requires_grad = True

    class_weights = compute_class_weights(train_labels).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01
    )

    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0
    training_log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) / GRAD_ACCUM
            loss.backward()

            total_train_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            del input_ids, attention_mask, labels, outputs
            free_memory()

        avg_train = total_train_loss / len(train_loader)
        val_loss, val_f1, val_auc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val Loss: {val_loss:.4f} | F1: {val_f1:.4f}")

        training_log.append({
            "horizon": horizon, "epoch": epoch, "train_loss": avg_train,
            "val_loss": val_loss, "val_f1": val_f1, "val_auc": val_auc
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            model.save_pretrained(champion_dir)
            AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(champion_dir)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Final Eval
    best_model = BertForSequenceClassification.from_pretrained(champion_dir, num_labels=2)
    best_model.to(DEVICE)
    _, test_f1, test_auc, test_preds, test_labels, test_probs = evaluate(best_model, test_loader, criterion, DEVICE)

    # Save results
    pd.DataFrame(training_log).to_csv(os.path.join(RESULTS_DIR, f'finbert_training_log_{horizon}.csv'), index=False)
    pd.DataFrame({
        'true_label': test_labels,
        'pred_label': test_preds,
        'prob_up': test_probs
    }).to_csv(os.path.join(RESULTS_DIR, f'finbert_test_predictions_{horizon}.csv'), index=False)

    with open(os.path.join(champion_dir, 'training_config.json'), 'w') as f:
        json.dump({
            'horizon': horizon, 'best_epoch': best_epoch, 'test_f1': test_f1, 'test_auc': test_auc
        }, f, indent=2)

    print(f'Training complete. Best F1: {test_f1:.4f}')

if __name__ == '__main__':
    main()