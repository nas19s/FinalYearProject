# Stock Drift Prediction using FinBERT and Hybrid Feature Engineering

## Project Overview

This final year project focuses on building and evaluating a machine learning pipeline capable of predicting short-term stock price drift (UP/DOWN classification) based on the textual content of corporate press releases, augmented by traditional financial features.

The core technical challenge lies in fine-tuning a specialised Transformer model (**FinBERT**) for financial sentiment analysis and demonstrating the performance and interpretability of this hybrid feature approach.

### 1. Project Scope 

To ensure feasibility and focus during the academic year, the project scope is strictly defined:

* **Universe:** Top 50 companies of the S&P 500 index.
* **Data Source:** Corporate Press Releases (Primary) and historical stock price data.
* **Time Horizon:** 10+ years of historical data.
* **Target Variable:** Short-term directional drift (UP/DOWN classification).

---

## 2. Technical Approach and Game Plan

The project is structured into four sequential phases, prioritising data feasibility and advanced model interpretability (SHAP analysis).

| Phase                                 | Goal                                                                                   | Key Deliverables                                               | Status              |
| :------------------------------------ | :------------------------------------------------------------------------------------- | :------------------------------------------------------------- | :------------------ |
| **Phase 1: Data Grind & Feasibility** | Lock down data sources, clean text, establish the initial baseline model (VADER + LR). | Working Data Pipeline, Baseline Metrics.                       | 🚧 In Progress (W1) |
| **Phase 2: Core CS Work**             | Implement, optimise, and rigorously test the advanced NLP model.                       | Fine-Tuned FinBERT Weights, Head-to-Head Evaluation Report.    | ☐ Planned           |
| **Phase 3: Analysis & Artefacts**     | Build interpretability (SHAP) and the practical demonstration tool (P&L Simulator).    | SHAP Visualisations (Token-Level), Streamlit Demo Application. | ☐ Planned           |
| **Phase 4: Polish & Defence**         | Finalise documentation and prepare for the academic inspection.                        | Final 10-Page Report, Code Polish, Inspection Practice.        | ☐ Planned           |

---

## 3. Repository Structure

This repository is organised to provide clear separation between code, data, documentation, and development notebooks, allowing for easy navigation during the project inspection.

| Folder             | Content                                                                                                                                   |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `01_Data/`         | Raw and processed data files (including the pilot test results). *(Note: Large, raw data files will be ignored by Git and not uploaded.)* |
| `02_Code/`         | All production Python scripts (scraper, model training, feature engineering, Streamlit app).                                              |
| `03_Notebooks/`    | Jupyter/Colab notebooks used for initial EDA, model experimentation, and SHAP analysis.                                                   |
| `04_Docs/`         | Project reports, scope documentation, and registration materials.                                                                         |
| `requirements.txt` | List of necessary Python packages to reproduce the environment.                                                                           |

---

## 4. Institutional Compliance Note (GitLab)

Initial development and version control are being tracked via this **GitHub** repository. This approach maximises efficiency during the early development phase. Upon receiving the official institutional GitLab repository link (expected Month X), the entire commit history will be **mirrored/imported** to ensure full compliance and preservation of the academic audit trail.
