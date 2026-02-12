import shap
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
TEXT_DIR = os.path.join(DATA_DIR, "processed_sec")
CSV_PATH = os.path.join(DATA_DIR, "final_feature_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "shap")

SAMPLE_SIZE = 5
TARGET_COL = 'Label_Month'

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_text(filename):
    path = os.path.join(TEXT_DIR, filename)
    if not os.path.exists(path): return ""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(1000) 

def main():
    print("Initializing SHAP Analysis...")

    device = torch.device("cpu") 
    
    try:
        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    # Prediction Pipeline
    def predict_pipe(texts):
        # FIX: Convert SHAP's numpy array to a standard list of strings
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]

        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            return torch.softmax(outputs.logits, dim=1).detach().numpy()

    # Load Data
    df = pd.read_csv(CSV_PATH)
    down_samples = df[df[TARGET_COL] == 0]
    
    if len(down_samples) < SAMPLE_SIZE:
        samples = down_samples
    else:
        samples = down_samples.sample(SAMPLE_SIZE, random_state=42)
    
    text_data = [load_text(f) for f in samples['filename']]
    text_data = [t for t in text_data if len(t) > 50] 
    
    print(f"Generating explanations for {len(text_data)} samples...")

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_pipe, masker)
    
    shap_values = explainer(text_data)

    print("Saving HTML visualization...")
    html_plot = shap.plots.text(shap_values[0], display=False)
    with open(os.path.join(RESULTS_DIR, "shap_explanation.html"), "w", encoding='utf-8') as f:
        f.write(html_plot)
    
    print("Saving summary bar chart...")
    plt.figure()
    shap.plots.bar(shap_values.mean(0), max_display=12, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar_summary.png"), bbox_inches='tight')

    print(f"SHAP Analysis Complete. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()