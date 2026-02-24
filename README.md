# Predicting Short-to-Medium Term Stock Drift using FinBERT & Hybrid Ensembles

## Project Overview

This final year project builds a machine learning pipeline to predict medium-term stock price drift (UP/DOWN classification) by combining advanced Natural Language Processing (NLP) with traditional technical analysis.

The core innovation is a **Hybrid Ensemble strategy**: 
1.  **Fine-tuning FinBERT** (a financial Transformer) on SEC filings to extract linguistic sentiment signals.
2.  **Fusing these signals** with **Textual Complexity metrics** (Gunning Fog Index) and **Market Momentum indicators** (RSI, MACD, Volume) to create a robust risk-management model.

### 1. Project Scope 

* **Universe:** Top 50 companies of the S&P 500 (historical & current).
* **Data Source:** SEC EDGAR (10-K, 10-Q) and Yahoo Finance price data.
* **Target Variable:** 1-Month Forward Returns (Classified as UP/DOWN based on median drift).
* **Methodology:** Comparative analysis between a Baseline (Logistic Regression + VADER), a Pure NLP Model (Fine-tuned FinBERT), and a Hybrid Ensemble (FinBERT + Technicals).

---

## 2. Technical Approach & Progress

The project follows a structured 4-phase development plan.

| Phase | Goal | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundations** | Build scraper, "Nuclear" HTML cleaning, and baseline. | • Cleaned SEC Dataset<br>• Baseline Model (54.8% Accuracy) |  **Complete** |
| **Phase 2: NLP Core** | Fine-tune FinBERT on weighted SEC labels. | • **Fine-Tuned FinBERT**<br>• Macro F1: 0.597 |  **Complete** |
| **Phase 3: Synthesis** | Boost performance via Hybrid Ensemble and SHAP analysis. | • **Hybrid RF Model**<br>• SHAP Token Attribution Dashboard | **Complete** |
| **Phase 4: Deployment** | Interactive visualization and strategic backtesting. | • **StockDrift Streamlit App**<br>• Event-Driven Backtest (Sharpe 0.84) |  **In Progress** |

---

## 3. Key Findings & Performance Benchmarks

### Model Comparison (Validation Set)
| Metric | Baseline (LogReg + VADER) | FinBERT (Fine-Tuned) |
| :--- | :--- | :--- |
| **Accuracy** | 54.8% | **62.4%** |
| **Macro F1** | 0.513 | **0.597** |
| **Down Recall** | 39.4% | **51.5%** |

### Hybrid Feature Importance (Random Forest)
When fusing NLP with Technicals, the model prioritizes:
1. **Gunning Fog Index** (Weight: 0.26): Textual complexity is a primary predictor of risk.
2. **Volume Change** (Weight: 0.21): Pre-filing liquidity signals.
3. **RSI** (Weight: 0.17): Momentum confirmation.
4. **FinBERT Score** (Weight: 0.14): Linguistic sentiment probability.

### Strategic Backtest (Sharpe 0.84)
Using a "Skip Top 25% Risk" strategy (avoiding filings predicted AS 'Down' by the model):
* **Mean Return:** 3.08% per trade.
* **Win Rate:** 66.7%.
* **Sharpe Ratio:** **0.841** (vs 0.626 for 'Buy All').
* **Total Return:** **349.9%** over the test horizon.

---

## 4. Error Analysis & Limitations

A rigorous audit of the model's 37.6% error rate revealed specific failure patterns:
* **Overconfidently Bullish Bias:** 9 out of 10 of the most confident errors were False Positives (predicted "Up" during a crash).
* **Complexity Trap:** Errors involve significantly more complex text (**Avg Gunning Fog: 39** vs 28 overall).
* **Technical Ambiguity:** The model struggles when **RSI is near neutral (40-60)**, lacking a strong technical anchor for the sentiment.

---

## 5. Interactive Dashboard (StockDrift App)

The project includes a production-ready Streamlit application for real-time inference and analysis.

**Features:**
*   Live Inspector: Paste raw SEC text to see real-time FinBERT predictions.
*   SHAP Attribution: Interactive visualization of exactly which words (e.g., "bottlenecks", "uncertainty") drove the model's decision.
*   Backtest Viewer: Comparative equity curves for Long-Only and Long-Short strategies.
*   Patterns: Real-time error cluster analysis and feature distributions.

**To run the app:**
```bash
python 05_App/app.py
```

---

## 6. Setup & Usage

**Prerequisites:**
* Python 3.10+
* Conda Environment: `stockdrift`

**Installation:**
```bash
conda activate stockdrift
pip install -r requirements.txt
```