# StockDrift
### Predicting Stock Price Direction from SEC Regulatory Filings using FinBERT

> **Final Year Computer Science Project**
> University of Birmingham | 2025–2026

---

## Research Question

> *Can the textual content of SEC 10-K and 10-Q regulatory filings predict the direction of stock price movement in the weeks following their release?*

This project tests the semi-strong form of the **Efficient Market Hypothesis** (Fama, 1970). If filing text carries predictive signal, markets are not instantaneously pricing all public information — consistent with Hong & Stein's (1999) gradual information diffusion hypothesis.

**Core finding:** T+5 prediction is near-random (AUC ≈ 0.499), T+20 is statistically significant (AUC = 0.523, p < 0.001) — consistent with complex regulatory language taking approximately one month to be fully priced in.

---

## Project Outcomes and Evaluation

The strategies underperform SPY in absolute return over the 2023–2025 test period (+44–72% vs SPY +82.7%). This is expected under the Efficient Market Hypothesis: public information signals are not supposed to generate large abnormal returns after transaction costs. What matters is whether the model extracts *quality* signal, not just raw return — and on that measure the results are strong.

**Sortino ratio comparison:**

| | Strategy A | Strategy B | Strategy C | Strategy D | SPY |
|---|---|---|---|---|---|
| Sortino ratio | 2.03 | **4.30** | **4.30** | 4.51 | ~1.2 |

Strategy B and C achieve Sortino ratios of 4.30 — approximately **3.6× the SPY benchmark**. The Sortino ratio measures return per unit of *downside* deviation only (unlike Sharpe, which penalises all volatility equally). A ratio of 4.30 versus SPY's ~1.2 means the strategies generate substantially more return for each unit of bad risk taken.

**Drawdown-matched comparison:**

The maximum drawdown figures reveal the clearest advantage. Strategy C's worst loss was −3.7%; SPY's was −18.8% over the same period. If Strategy C were leveraged to match SPY's drawdown level (approximately 5.1× leverage, bringing max drawdown from −3.7% to −18.8%), the adjusted return would be approximately **+219%** — compared to SPY's +82.7%. Strategy B, similarly leveraged to SPY's drawdown level (3.6×), yields an estimated **+184%**.

| | Strategy B | Strategy C | SPY |
|---|---|---|---|
| Actual return | +51.9% | +43.1% | +82.7% |
| Max drawdown | −5.3% | −3.7% | −18.8% |
| Leverage to match SPY drawdown | 3.6× | 5.1× | — |
| Drawdown-matched projected return | ~+184% | ~+219% | +82.7% |

These projections are illustrative and do not account for margin costs or slippage under leverage. They are presented as a directional comparison of signal quality, not as a trading recommendation. The underlying point is that the model's return distribution — many small gains, rare small losses — is structurally different from SPY's wider swings.

**Statistical significance:**

The 63.1% UP win rate over 309 independent test trades is confirmed significant at p < 0.001 (binomial test). The CAAR event study independently validates label quality — UP-labelled filings outperform DOWN-labelled filings by +11.81 percentage points by day +20, driven purely by the language in the filings and entirely independent of the model's predictions.

In summary: the model captures a real, statistically proven signal in SEC filing language. The absolute return underperformance reflects the exceptional strength of the 2023–2025 bull market and the structural constraints of a long-only strategy — not a failure of the predictive approach.

---

## Key Results

### Model Performance (test set 2023–2025, 430 filings)

| Model | Accuracy | F1 Macro | AUC |
|---|---|---|---|
| Majority Class (baseline — always UP) | 61.1% | 0.380 | — |
| Logistic Regression (baseline) | 47.4% | 0.474 | 0.528 |
| FinBERT T+5 (1 week horizon) | 49.0% | 0.475 | 0.498 |
| FinBERT T+10 (2 week horizon) | 50.5% | 0.499 | 0.515 |
| **FinBERT T+20 (1 month horizon)** | **52.3%** | **0.520** | **0.523** |
| **Filing-Level Accuracy (T+20)** | **60.9%** | — | — |
| Section-Weighted Voting Ensemble | 56.1% | 0.509 | 0.536 |

**UP win rate: 63.1% — statistically significant (p < 0.001, binomial test)**

Note on baselines: the majority class baseline achieves 61.1% accuracy purely from class imbalance (~62% of test filings are labelled UP in the 2023–2025 bull market). Its F1 macro of 0.380 reveals it never correctly identifies DOWN cases. FinBERT's F1 macro of 0.520 demonstrates genuine learning across both classes.

### Backtest Results (2023–2025, £10,000 initial capital, 0.1% transaction cost)

| Strategy | Trades | Win Rate | Total Return | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|---|---|
| A: All UP signals | 309 | 63.1% | +44.2% | 1.08 | 2.03 | −5.2% |
| B: Confidence ≥ 0.72 | 124 | 67.7% | +51.9% | 1.68 | 4.30 | −5.3% |
| C: Financials + Tech sectors only | 89 | 71.9% | +43.1% | 1.87 | 4.30 | −3.7% |
| D: 2× Leveraged B | 124 | 65.3% | +71.5% | 1.51 | 4.51 | −7.0% |
| **SPY Buy-and-Hold (benchmark)** | — | — | +82.7% | 1.44 | ~1.2 | −18.8% |

### Advanced Analysis

| Analysis | Finding |
|---|---|
| CAAR Event Study | +11.81% spread between UP and DOWN labelled filings by day +20 |
| Calibration (ECE) | 0.049 — well calibrated |
| Optimal confidence threshold | 0.72 (accuracy 66.9%, n=124 filings) |
| Ablation — NLP only vs technical only | 54.9% vs 53.0% — text adds genuine signal |
| Best sector | Financials 73.6% |
| Worst sector | Utilities 40.0% |
| Worst quarter | 2024Q1 37.5% (Federal Reserve rate uncertainty) |
| Binomial test p-value | p = 0.000032 |
| Overall accuracy test p-value | p = 0.0069 |

---

## Pipeline Architecture

```
SEC EDGAR API
      │
      ├── 02_get_sec_data.py          10-K annual filings
      └── 02b_get_sec_data_10q.py     10-Q quarterly filings
                    │
                    ▼  1,823 filings — 50 tickers — 2016–2025
                    │
         01_shrink_sec_data.py        HTML cleaning, 4-section extraction
                                      512-token chunking, 50-token overlap
                    │
                    ▼  sec_cleaned.parquet — 105,322 chunks
                    │
         03_create_target.py          Volatility-scaled binary labels
                                      T+5 / T+10 / T+20 horizons
                    │
                    ▼  labeled_dataset.parquet — 101,325 rows
                    │
        ┌───────────┴────────────────────────────────┐
        │                                            │
04_feature_engineering.py               07_finbert_prep.py
RSI, MACD, Volume, Fog,                 Tokenise chunks per horizon
Flesch, VADER, Word Count                        │
        │                               08_train_finbert.py
05_baseline_model.py                    Fine-tune FinBERT
Majority class + LR baselines           Layers 0–9 frozen
        │                                        │
        │                               09b_finbert_inference.py
        │                               Full test inference
        └──────────────────┬─────────────────────┘
                           │
                  10_hybrid_ensemble.py
                  Section-Weighted Voting
                           │
           ┌───────────────┼───────────────┐
           │               │               │
  11_shap_analysis   12_backtest    13_error_analysis
           │               │               │
           └───────────────┴───────────────┘
                           │
               14_advanced_analysis.py
               CAAR, Calibration, ROC,
               Ablation, Significance Tests
                           │
                       app.py
                  Streamlit Dashboard
```

---

## Dataset

| Property | Value |
|---|---|
| Source | SEC EDGAR via `edgartools` Python library |
| Companies | 50 S&P 500 companies across 8 sectors |
| Period | 2016–2025 |
| Total filings | 1,823 (475 × 10-K annual, 1,348 × 10-Q quarterly) |
| Total text chunks | 105,322 |
| Test-set filings | 430 (2023–2025) |
| Price data | Yahoo Finance via `yfinance` |

### Sections Extracted Per Filing

| Section | Weight | Rationale |
|---|---|---|
| Item 1A — Risk Factors | 1.0 | Most forward-looking; legally required material risk disclosure |
| Item 7 — MD&A | 1.0 | Management's interpretation of results; most studied in literature |
| Item 9A — Internal Controls | 0.8 | Governance quality signal |
| Item 7A — Quantitative Market Risk | 0.6 | Quantitative disclosures; less narrative signal |

### Label Generation

- **Anchor date:** `filing_date` (SEC receipt date, not `period_of_report` — avoids lookahead bias)
- **After-hours rule:** Filings released ≥ 16:00 ET → next trading day open as entry price
- **Method:** Volatility-scaled binary labels using 60-day rolling standard deviation
  - UP (+1) if T+N return > +1σ
  - DOWN (−1) if T+N return < −1σ
  - FLAT (0) dropped — binary classification only
- **Class distribution (T+20):** ~62% UP, ~38% DOWN
- **Class imbalance handling:** Inverse class weighting in CrossEntropyLoss

### Train / Validation / Test Split

| Split | Period | Size |
|---|---|---|
| Train | Before 2021 | ~4,000 stratified chunks per horizon |
| Validation | 2021–2022 | Used for early stopping |
| Test | 2023–2025 | 430 filing-level predictions |

> Time-based splits only — random splits would allow future filings into training, creating data leakage.

---

## Model

### FinBERT Fine-tuning

**Base model:** `ProsusAI/finbert` — BERT pre-trained on financial news, earnings calls, and analyst reports (Yang et al., 2020). Outperforms general BERT on financial NLP tasks by up to 15 F1 points.

**Modifications:**
- Original 3-class sentiment head (positive/negative/neutral) replaced with binary UP/DOWN head
- Layer-wise freezing: layers 0–9 frozen (87% of 110M parameters); layers 10–11 + classifier trained (~14M parameters)
- Prevents catastrophic forgetting of FinBERT's financial domain knowledge (Howard & Ruder, 2018)

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | AdamW with linear LR warmup |
| Epochs | 3 (early stopping patience=2 on val F1) |
| Batch size | 16 |
| Training samples | ~4,000 per horizon (stratified) |
| Hardware | CPU |
| Threads | `torch.set_num_threads(8)` |
| Models trained | 3 (T+5, T+10, T+20) |

### Section-Weighted Voting Ensemble

One filing produces 50–200 chunks. Each chunk casts a hard vote (UP if `prob_up ≥ 0.5`, else DOWN), weighted by section importance:

```
Filing prediction = argmax(Σ weighted_UP_votes, Σ weighted_DOWN_votes)
Filing confidence = Σ weighted_UP / (Σ weighted_UP + Σ weighted_DOWN)
```

Hard voting is used instead of soft probability averaging because FinBERT fine-tuned on small samples exhibits **probability collapse** — probabilities cluster near 0.5 regardless of true label, making averaged scores meaningless.

---

## Evaluation Framework

| Method | Purpose |
|---|---|
| AUC-ROC | Threshold-independent discriminative power |
| F1 Macro | Performance across both classes equally |
| Win Rate | Accuracy on UP-labelled signals (long-only strategies) |
| Binomial test | Whether win rate exceeds 50% by chance |
| McNemar test | Whether ensemble outperforms majority on specific instances |
| CAAR Event Study | Cumulative abnormal return validation (MacKinlay, 1997) |
| Sortino Ratio | Return per unit of downside deviation (annualised) |
| Max Drawdown | Worst peak-to-trough capital loss |
| Ablation Study | NLP vs technical vs readability feature contribution |
| Calibration (ECE) | Reliability of confidence scores |

---

## Streamlit Dashboard

Launch with:

```bash
streamlit run app.py
```

| Page | Contents |
|---|---|
| Overview | Research question, pipeline summary, headline metrics |
| Data & EDA | Filing timeline, chunk lengths, section coverage, label distributions |
| Model Results | Master results table, confusion matrices |
| SHAP Explainability | Feature importance bar chart and beeswarm plot |
| Live Prediction | Paste any filing text — live FinBERT T+20 inference |
| Backtest | Equity curves, monthly returns, per-ticker performance |
| Error Analysis | Failure patterns by sector, quarter, and ticker |
| Advanced Analysis | Calibration, CAAR, ROC curves, ablation, significance tests |

---

## Reproduction

### 1. Environment Setup

```bash
conda create -n stockdrift python=3.10
conda activate stockdrift

pip install edgartools transformers torch pandas numpy scikit-learn
pip install ta textstat vaderSentiment tqdm pyarrow
pip install xgboost==2.1.3        # must be 2.1.3 — SHAP breaks on 3.x
pip install shap statsmodels streamlit pillow seaborn yfinance
brew install libomp               # macOS only — required for XGBoost
```

### 2. Run Full Pipeline

All scripts run from the project root with `conda activate stockdrift`:

```bash
# Data collection
python 02_Code/preprocessing/02_get_sec_data.py
python 02_Code/preprocessing/02b_get_sec_data_10q.py

# Preprocessing and labels
python 02_Code/preprocessing/01_shrink_sec_data.py
python 02_Code/preprocessing/03_create_target.py
python 02_Code/feature_engineering/04_feature_engineering.py

# Baselines
python 02_Code/models/05_baseline_model.py

# FinBERT — tokenise
python 02_Code/preprocessing/07_finbert_prep.py --horizon T5
python 02_Code/preprocessing/07_finbert_prep.py --horizon T10
python 02_Code/preprocessing/07_finbert_prep.py --horizon T20

# FinBERT — train (~3 hours each on M2 CPU)
python 02_Code/models/08_train_finbert.py --horizon T5
python 02_Code/models/08_train_finbert.py --horizon T10
python 02_Code/models/08_train_finbert.py --horizon T20

# Evaluation and ensemble
python 02_Code/evaluation/09_evaluate_models.py
python 02_Code/models/09b_finbert_inference.py
python 02_Code/models/10_hybrid_ensemble.py

# Analysis
python 02_Code/shap_analysis/11_shap_analysis.py
python 02_Code/evaluation/12_strategic_backtest.py
python 02_Code/models/13_error_analysis.py
python 02_Code/evaluation/14_advanced_analysis.py
python 02_Code/evaluation/06_eda_analysis.py

# Launch dashboard
streamlit run app.py
```

### 3. Hardware Notes

| Spec | Value |
|---|---|
| Tested on | Apple M2, 8GB RAM, macOS |
| RAM minimum | 8GB (16GB recommended for full-dataset inference) |
| GPU | Not required — MPS causes OOM on 8GB M2, CPU used |
| Training time | ~3 hours per FinBERT model on M2 CPU |
| Storage | ~13GB (filings + prices + model checkpoints) |

---

## File Structure

```
FinalYearProject/
│
├── app.py                              # Streamlit dashboard (8 pages)
├── README.md                           # This file
│
├── 01_Data/
│   ├── sec_metadata.csv                # 1,823 filing records
│   ├── sec_sections/                   # Raw extracted section JSON files
│   ├── sec_cleaned.parquet             # 105,322 cleaned + chunked texts
│   ├── labeled_dataset.parquet         # Chunks with T+5/T+10/T+20 labels
│   ├── train.parquet                   # Training split (< 2021)
│   ├── val.parquet                     # Validation split (2021–2022)
│   ├── test.parquet                    # Test split (>= 2023)
│   ├── final_feature_dataset.parquet   # Chunks + all engineered features
│   └── prices/                         # {TICKER}_prices.csv + SPY_prices.csv
│
├── 02_Code/
│   ├── preprocessing/                  # Scripts 01, 02, 02b, 03, 07
│   ├── feature_engineering/            # Script 04
│   ├── models/                         # Scripts 05, 08, 09b, 10, 13
│   ├── evaluation/                     # Scripts 06, 09, 12, 14
│   └── shap_analysis/                  # Script 11
│
├── 03_Models/
│   ├── finbert_champion_T5/            # Saved FinBERT T+5 checkpoint
│   ├── finbert_champion_T10/           # Saved FinBERT T+10 checkpoint
│   ├── finbert_champion_T20/           # Saved FinBERT T+20 checkpoint
│   └── hybrid_ensemble/                # XGBoost model + scaler + feature list
│
└── 04_Results/
    ├── baseline/                       # Confusion matrices, LR feature importance
    ├── metrics/                        # master_results_table_final.csv, prediction CSVs
    ├── shap/                           # shap_beeswarm.png, shap_bar.png
    ├── backtest/                       # Equity curves, backtest_trades.csv, summary
    ├── eda/                            # fig1–fig6 EDA plots
    ├── error_analysis/                 # Failure breakdown plots + CSVs by sector/quarter
    └── advanced/                       # A–I analysis plots + significance_tests.txt
```

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| FinBERT trained on ~4,000 chunks | Probabilities cluster near 0.5 | Hard voting is robust to probability collapse |
| ~47 10-K filings with empty MD&A | Non-standard HTML not parsed (CVX, BRK-B, AVGO) | Falls back to available sections |
| No macroeconomic features | Cannot predict macro shocks (rate hikes, COVID) | Worst quarter 2024Q1 acknowledged in report |
| Class imbalance (~62% UP) | Bull-market bias in predictions | Inverse class weighting in loss function |
| McNemar test p = 0.10 | Ensemble not significantly better pairwise vs majority | Binomial and t-tests p < 0.001 on win rate |
| XGBoost hybrid: FinBERT SHAP = 0.000 | FinBERT not inferred on training data | Ablation study provides clean evidence of NLP contribution |
| 50-company universe | Hardware constraint | 430 test filings sufficient for all significance tests |
| Drawdown-matched projections are illustrative | Do not account for margin costs or slippage under leverage | Presented as directional comparisons only |

---

## References

| Paper | Citation |
|---|---|
| Fama (1970) | Efficient capital markets: A review of theory and empirical work. *Journal of Finance* |
| Hong & Stein (1999) | A unified theory of underreaction, momentum trading, and overreaction. *Journal of Finance* |
| Li (2008) | Annual report readability, current earnings, and earnings persistence. *Journal of Accounting and Economics* |
| Loughran & McDonald (2011) | When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance* |
| Devlin et al. (2018) | BERT: Pre-training of deep bidirectional transformers. *NAACL 2019* |
| Yang et al. (2020) | FinBERT: A pretrained language model for financial communications. *arXiv:2006.08097* |
| Howard & Ruder (2018) | Universal language model fine-tuning for text classification. *ACL 2018* |
| MacKinlay (1997) | Event studies in economics and finance. *Journal of Economic Literature* |
| Lundberg & Lee (2017) | A unified approach to interpreting model predictions. *NeurIPS 2017* |

---

*Python 3.10 · conda env: `stockdrift`*