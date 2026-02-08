# Predicting Short-to-Medium Term Stock Drift using FinBERT & Hybrid Ensembles

## Project Overview

This final year project builds a machine learning pipeline to predict medium-term stock price drift (UP/DOWN classification) by combining advanced Natural Language Processing (NLP) with traditional technical analysis.

The core innovation is a **Hybrid Ensemble strategy**: 
1.  **Fine-tuning FinBERT** (a financial Large Language Model) on SEC filings to extract linguistic sentiment signals.
2.  **Fusing these signals** with market momentum indicators (RSI, MACD, Volume) to create a robust risk-management model.

### 1. Project Scope 

* **Universe:** Top 50 companies of the S&P 500.
* **Data Source:** SEC Filings (8-K, 10-K, 10-Q) and historical price data.
* **Target Variable:** Medium-term directional drift (20-Day Lookahead).
* **Methodology:** Comparative analysis between a Baseline (Logistic Regression), a Pure NLP Model (FinBERT), and a Hybrid Ensemble (FinBERT + Technicals).

---

## 2. Technical Approach & Progress

The project follows a structured 4-phase development plan.

| Phase | Goal | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Data & Feasibility** | Build scraper, clean text ("Nuclear" method), and establish a baseline. | • Cleaned SEC Dataset<br>• Baseline Model (65% Accuracy, biased)<br>• Multi-horizon target definition |  **Complete** |
| **Phase 2: Core CS Work** | Fine-tune FinBERT to detect downturns and fix class imbalance. | • Class Weight Optimization<br>• Fine-Tuned FinBERT Model<br>• Head-to-Head Eval (FinBERT vs Baseline) |  **Complete** |
| **Phase 3: Analysis & Artifacts** | Boost performance via Hybrid Ensemble and explain results (SHAP). | • **Hybrid Model** (Tripled recall on drops)<br>• SHAP "Why?" Visualization<br>• Strategic Backtest (P&L Simulation) | **In Progress** |
| **Phase 4: Polish & Defense** | Finalize documentation and prepare for academic inspection. | • Final Report<br>• Code Refactoring<br>• Mock Defense | ☐ Planned |

---

## 3. Key Findings (So Far)

* **The Baseline Trap:** Standard models (Logistic Regression) achieved 65% accuracy but failed to predict *any* market drops (Recall: 0.00), rendering them useless for risk management.
* **FinBERT's Value:** The fine-tuned Transformer successfully learned to identify negative linguistic cues, improving the detection of downturns.
* **The Hybrid Advantage:** Combining FinBERT's text probability with **Volume** and **RSI** tripled the model's ability to predict stock drops (Recall: ~0.30) compared to using text alone.

---

## 4. Repository Structure

| Folder | Content |
| :--- | :--- |
| `01_Data/` | `processed_sec/` (Cleaned text), `prices/` (Market data), and `final_feature_dataset.csv`. |
| `02_Code/` | Production scripts:<br>• `scraper/`: Data collection.<br>• `preprocessing/`: Text cleaning & BERT tokenization.<br>• `models/`: Training scripts for Baseline, FinBERT, and Hybrid Ensemble.<br>• `evaluation/`: ROC curves, Confusion Matrices, and SHAP analysis. |
| `03_Models/` | Saved model weights (including the Champion FinBERT model). |
| `04_Results/` | Generated artifacts: SHAP HTML plots, Confusion Matrices, and Feature Importance logs. |
| `04_Docs/` | Project scope documentation and academic reports. |

---

## 5. Setup & Usage

**Prerequisites:**
* Python 3.10+
* PyTorch (MPS enabled for Mac / CUDA for NVIDIA)

**Installation:**
```bash
pip install -r requirements.txt