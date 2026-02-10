import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
import datetime
import gc # Garbage Collector for memory management

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data", "finbert_tensors")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")

# --- LOW MEMORY SETTINGS (For 8GB RAM) ---
BATCH_SIZE = 2          # Process only 2 samples at a time
GRAD_ACCUMULATION = 4   # Update model every 4 steps (Effective Batch = 8)
EPOCHS = 3              # Reduce to 3 epochs to finish faster
LEARNING_RATE = 2e-5

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def load_dataset(split_name):
    path = os.path.join(DATA_DIR, f"{split_name}.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path)
    return TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])

def train():
    # Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU (Slow but safe).")

    # Load Data
    train_dataset = load_dataset("train")
    val_dataset = load_dataset("val")
    
    if not train_dataset:
        print("Error: Training data not found.")
        return

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    
    # Adjust steps for gradient accumulation
    total_steps = len(train_dataloader) // GRAD_ACCUMULATION * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Load Weights
    weights_path = os.path.join(DATA_DIR, "class_weights.pt")
    if os.path.exists(weights_path):
        class_weights = torch.load(weights_path).to(device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0
    start_time = time.time()

    print(f"Starting training (Low Memory Mode: Batch {BATCH_SIZE})...")

    for epoch_i in range(0, EPOCHS):
        print(f"\nEpoch {epoch_i + 1} / {EPOCHS}")
        
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            # Move to device
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            loss = loss_fct(outputs.logits, b_labels)
            
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUMULATION
            total_train_loss += loss.item()
            loss.backward()

            # Update weights only after accumulating enough gradients
            if (step + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # --- MEMORY CLEANUP ---
                if step % 50 == 0:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        avg_train_loss = total_train_loss / len(train_dataloader) * GRAD_ACCUMULATION
        print(f"Average Training Loss: {avg_train_loss:.2f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(b_labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(MODEL_SAVE_DIR)
            print("New best model saved.")

    print(f"\nFine-tuning complete. Total time: {format_time(time.time()-start_time)}")

if __name__ == "__main__":
    train()