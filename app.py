"""
app.py — StockDrift Research Dashboard
AI tools were used only for minor documentation wording and comment refinements and Beautifiying Pages and desgin.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS      = os.path.join(PROJECT_ROOT, "04_Results")
DATA_DIR     = os.path.join(PROJECT_ROOT, "01_Data")

st.set_page_config(
    page_title="StockDrift — SEC Filing Prediction",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Georgia', 'Times New Roman', serif; }
    h1 { font-size: 1.9rem !important; font-weight: 700; letter-spacing: -0.02em; }
    h2 { font-size: 1.3rem !important; font-weight: 600; color: #1a1a2e; }
    h3 { font-size: 1.1rem !important; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; padding: 4px 0; }
    div[data-testid="metric-container"] { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px 18px; }
    div[data-testid="metric-container"] label { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; color: #666 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-family: 'Courier New', monospace; color: #1a1a2e !important; }
    hr { border: none; border-top: 1px solid #e8e8e8; margin: 1.5rem 0; }
    div[data-testid="stInfo"] { background-color: #f0f4ff; border-left: 3px solid #3a5bd9; border-radius: 0 4px 4px 0; font-size: 0.9rem; }
    div[data-testid="stSuccess"] { background-color: #f0fff4; border-left: 3px solid #38a169; border-radius: 0 4px 4px 0; font-size: 0.9rem; }
    div[data-testid="stWarning"] { background-color: #fffbeb; border-left: 3px solid #d69e2e; border-radius: 0 4px 4px 0; font-size: 0.9rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #e0e0e0; border-radius: 4px; }
    div[data-testid="stCaptionContainer"] { color: #888; font-size: 0.8rem; font-style: italic; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def load_image(path):
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return Image.open(path)
    return None

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## StockDrift")
st.sidebar.markdown("SEC Filing → Price Direction")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigate the project**")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data & EDA",
        "Model Results",
        "SHAP Explainability",
        "Live Prediction",
        "Backtest",
        "Custom Backtest",
        "Error Analysis",
        "Advanced Analysis",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Sidebar navigation guide — helps professor know what to expect on each page
nav_guide = {
    "Overview":           "Project summary, research question, core results",
    "Data & EDA":         "1,823 filings, chunking, label generation",
    "Model Results":      "FinBERT T+5/T+10/T+20, baselines, training logs",
    "SHAP Explainability":"Feature importance — what the model learned",
    "Live Prediction":    "Paste filing text → live T+20 prediction",
    "Backtest":           "4 strategies vs SPY benchmark",
    "Custom Backtest":    "Interactive engine — filter, adjust, simulate",
    "Error Analysis":     "Where and why the model fails",
    "Advanced Analysis":  "CAAR, ROC, ablation, significance tests",
}
if page in nav_guide:
    st.sidebar.caption(nav_guide[page])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Final Year CS Project<br>University of Birmingham<br>2025–2026<br><br>"
    "Supervisor: Dr Jianbo Jiao</small>",
    unsafe_allow_html=True,
)

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","GOOGL":"Technology",
    "META":"Technology","AVGO":"Technology","AMD":"Technology","ORCL":"Technology",
    "CRM":"Technology","INTC":"Technology",
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","BLK":"Financials","AXP":"Financials","SCHW":"Financials",
    "USB":"Financials","BRK-B":"Financials",
    "JNJ":"Healthcare","UNH":"Healthcare","LLY":"Healthcare","PFE":"Healthcare",
    "ABBV":"Healthcare","MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare",
    "DHR":"Healthcare","BMY":"Healthcare",
    "AMZN":"Consumer","TSLA":"Consumer","HD":"Consumer","MCD":"Consumer",
    "NKE":"Consumer","SBUX":"Consumer","TGT":"Consumer","COST":"Consumer",
    "BA":"Industrials","CAT":"Industrials","HON":"Industrials","UPS":"Industrials",
    "RTX":"Industrials","DE":"Industrials",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "LIN":"Materials","APD":"Materials","NEM":"Materials",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
}


# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("StockDrift")
    st.markdown("**Predicting Stock Price Direction from SEC Regulatory Filings using FinBERT**")
    st.markdown("*Nasrudin Adan · University of Birmingham · 2025–2026 · Supervisor: Dr Jianbo Jiao*")
    st.markdown("---")

    st.markdown("""
    > **Research Question:** Can the textual content of SEC 10-K and 10-Q regulatory filings
    > predict the direction of stock price movement in the weeks following their release?
    """)

    st.markdown(
        "SEC filings are legally mandated corporate disclosures — management can face criminal "
        "charges for lying in them. A single 10-K can exceed 100,000 words. No human analyst "
        "can read thousands simultaneously. This project tests whether a fine-tuned language "
        "model can extract actionable predictive signal from these documents at scale."
    )
    st.markdown(
        "The core theoretical question is whether the "
        "**semi-strong Efficient Market Hypothesis** (Fama, 1970) holds for SEC filings: "
        "if filing text carries predictive signal, markets are not immediately pricing all "
        "public information. The finding that T+5 is near-random but T+20 is significant "
        "supports Hong & Stein's (1999) **gradual information diffusion** hypothesis — "
        "complex regulatory language takes approximately one month to be fully priced in."
    )

    st.markdown("---")
    st.subheader("Dataset at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filings collected", "1,823", "475 × 10-K, 1,348 × 10-Q")
    col2.metric("S&P 500 tickers",   "50",    "8 sectors, 2016–2025")
    col3.metric("Text chunks",        "105,322","512 tokens, 50-token overlap")
    col4.metric("Test-set filings",   "430",   "2023–2025, held-out")

    st.markdown("---")
    st.subheader("Core Finding — The Horizon Effect")

    col1, col2, col3 = st.columns(3)
    col1.metric("T+5 AUC  (1 week)",  "0.498", "≈ random — markets react fast")
    col2.metric("T+10 AUC (2 weeks)", "0.515", "+0.017 above random")
    col3.metric("T+20 AUC (1 month)", "0.523", "+0.025, p < 0.001")

    st.info(
        "**Why does the horizon matter?** T+5 being near-random is expected — "
        "institutional investors and algorithms reprice obvious signals within days. "
        "T+20 being significant (p < 0.001) means the full informational content of complex "
        "regulatory language takes approximately one month to be absorbed by the market. "
        "This is the central empirical finding of the project."
    )

    st.markdown("---")
    st.subheader("Key Results")

    results = load_csv(os.path.join(RESULTS, "metrics", "master_results_table_final.csv"))
    if results is not None:
        st.dataframe(
            results.style.highlight_max(
                subset=[c for c in ["F1_Macro","AUC","Accuracy"] if c in results.columns],
                color="#d4edda",
            ),
            use_container_width=True,
        )
        st.caption(
            "Green highlights show the best value per metric column. "
            "Note that the Majority Class baseline achieves 61.1% accuracy purely from class imbalance "
            "(62% of filings are labelled UP in the 2023–2025 bull market) — its F1 Macro of 0.380 "
            "reveals it never identifies DOWN cases. F1 Macro is the honest metric here."
        )
    else:
        st.caption("Run 09_evaluate_models.py to generate the master results table.")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.markdown("The project implements a complete end-to-end pipeline from raw SEC filings to backtested trading strategies:")
    st.code("""
SEC EDGAR API  →  Raw Filings (10-K / 10-Q)
                          |
          HTML Cleaning + Stride-128 Chunking (512 tokens)
                          |
      Volatility-Scaled Label Generation  (T+5 / T+10 / T+20)
                          |
        ┌─────────────────┼──────────────────┐
        |                 |                  |
FinBERT Fine-tuning   RSI / MACD /       Baselines
(layers 10-11 only)   NLP Features    (LR, Majority)
        |                 |
 Filing-Level Probs   Feature Dataset
        └─────────────────┘
                  |
    Section-Weighted Voting Ensemble
                  |
      Backtest + SHAP Analysis + CAAR Event Study
    """, language="text")
    st.caption(
        "FinBERT layers 0–9 are frozen (87% of parameters unchanged) to preserve pre-trained "
        "financial language knowledge. Only layers 10–11 and the classification head are trained "
        "— approximately 14M of 110M parameters. This prevents catastrophic forgetting "
        "(Howard & Ruder, 2018)."
    )


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data & EDA":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Data & Exploratory Analysis")
    st.markdown("---")

    st.markdown(
        "This page documents the dataset construction. Understanding the data decisions — "
        "which sections were extracted, how labels were computed, why time-based splits — "
        "is essential context for evaluating the model results."
    )

    st.markdown("---")
    st.subheader("Dataset Statistics")
    meta_path = os.path.join(DATA_DIR, "sec_metadata.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total filings",  f"{len(meta):,}")
        col2.metric("10-K annual",    f"{(meta['form_type']=='10-K').sum():,}")
        col3.metric("Unique tickers", f"{meta['ticker'].nunique():,}")
        col4.metric("10-Q quarterly", f"{(meta['form_type']=='10-Q').sum():,}")
    else:
        st.info("sec_metadata.csv not found — run 02_get_sec_data.py to populate.")

    st.markdown("---")
    st.subheader("Section Extraction Weights")
    st.markdown(
        "Four sections were extracted from each filing using HTML boundary detection. "
        "Section weights determine how much each section's chunk votes count toward the "
        "final filing-level prediction. Weights are grounded in prior literature:"
    )
    weight_df = pd.DataFrame({
        "Section":   ["Item 1A — Risk Factors","Item 7 — MD&A","Item 9A — Internal Controls","Item 7A — Market Risk"],
        "Weight":    [1.0, 1.0, 0.8, 0.6],
        "Why This Weight": [
            "Most forward-looking section; legally required material risk disclosure (weight 1.0)",
            "Management's own interpretation of results; most studied in academic literature — Li (2008) (weight 1.0)",
            "Governance quality signal; procedural but informative (weight 0.8)",
            "Quantitative market risk disclosures; less narrative, more formulaic content (weight 0.6)",
        ],
    })
    st.dataframe(weight_df, use_container_width=True, hide_index=True)
    st.info(
        "**Why not weight all sections equally?** Li (2008) showed MD&A is the most predictive "
        "section for future earnings. Loughran & McDonald (2011) showed Risk Factors language "
        "predicts future volatility. Market Risk (Item 7A) contains mostly quantitative tables "
        "with less forward-looking narrative — hence the lower weight."
    )

    st.markdown("---")
    st.subheader("Label Generation")
    st.markdown(
        "Labels were generated using **volatility-scaled thresholds** — a 2% move means "
        "something different for Tesla versus Johnson & Johnson. Each stock's own 60-day "
        "rolling standard deviation is used as the threshold:"
    )
    label_df = pd.DataFrame({
        "Label": ["UP (+1)", "DOWN (−1)", "FLAT (0)"],
        "Condition": [
            "T+N return > +1 rolling standard deviation",
            "T+N return < −1 rolling standard deviation",
            "Return within ±1 standard deviation — dropped",
        ],
        "Why": [
            "Meaningful positive move relative to each stock's own volatility",
            "Meaningful negative move relative to each stock's own volatility",
            "Ambiguous outcome — not useful for binary classification",
        ],
    })
    st.dataframe(label_df, use_container_width=True, hide_index=True)
    st.warning(
        "**Critical methodological decision — filing_date not period_of_report:** "
        "Labels are anchored to the date the filing was publicly released, not when the "
        "accounting period ended. A 10-Q for Q3 (ending September 30) might not be filed "
        "until November. Using period_of_report would give the model knowledge of the future "
        "— this is lookahead bias. filing_date is the correct anchor."
    )

    st.markdown("---")
    st.subheader("Train / Validation / Test Splits")
    st.markdown(
        "**Time-based splits only** — no random shuffling. Random splits would allow a "
        "filing from 2024 to appear in training while a 2019 filing is in the test set, "
        "effectively giving the model knowledge of the future."
    )
    split_df = pd.DataFrame({
        "Split":       ["Train", "Validation", "Test"],
        "Period":      ["Before 2021", "2021–2022", "2023–2025"],
        "Size":        ["~4,000 stratified chunks per horizon", "Used for early stopping on val F1", "430 filing-level predictions"],
        "Purpose":     ["Fine-tune FinBERT weights", "Prevent overfitting via early stopping", "All reported results — held-out, never seen during training"],
    })
    st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("EDA Figures")
    st.markdown("The figures below were generated by running `06_eda_analysis.py`.")

    eda_path = os.path.join(RESULTS, "eda")
    eda_figures = [
        ("fig1_label_distributions.png",
         "Label Distributions (T+5, T+10, T+20)",
         "Shows the proportion of UP, DOWN, and FLAT labels at each horizon. "
         "The ~62% UP class imbalance is visible here — this is why F1 Macro is used "
         "as the primary metric rather than accuracy. Inverse class weighting in "
         "CrossEntropyLoss addresses this during training."),
        ("fig2_filing_timeline.png",
         "Filing Timeline by Year",
         "Distribution of 10-K and 10-Q filings across 2016–2025. "
         "The dataset spans diverse market conditions: 2016–2019 bull market, "
         "COVID crash 2020, recovery 2021–22, high interest rate period 2023–25."),
        ("fig3_chunk_lengths.png",
         "Chunk Token Length Distribution",
         "Most chunks hit the 512-token BERT architectural limit, confirming the "
         "filing sections are dense with content. The 50-token overlap between chunks "
         "preserves context at boundaries."),
        ("fig4_section_coverage.png",
         "Section Coverage per Filing",
         "Proportion of filings where each section was successfully extracted. "
         "Note that ~47 10-K filings (CVX, BRK-B, AVGO) had empty MD&A due to "
         "non-standard HTML — the pipeline falls back to available sections."),
        ("fig5_labels_by_form_type.png",
         "Labels by Form Type (10-K vs 10-Q)",
         "Compares UP/DOWN distribution between annual and quarterly filings. "
         "10-Qs are more numerous (1,348 vs 475) and capture quarterly momentum signals."),
        ("fig6_top_tickers.png",
         "Top 20 Tickers by Filing Count",
         "Companies with the most filings in the dataset. "
         "Companies with long histories and consistent filing schedules "
         "contribute more training signal."),
    ]

    any_shown = False
    for fname, title, explanation in eda_figures:
        img = load_image(os.path.join(eda_path, fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.caption(explanation)
            st.markdown("---")
            any_shown = True
    if not any_shown:
        st.info("No EDA figures found. Run 06_eda_analysis.py to generate them.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Model Results")
    st.markdown("---")

    st.markdown(
        "This page shows the performance of all models on the 430 held-out test filings "
        "(2023–2025). Results are reported at both **chunk level** (individual 512-token "
        "windows) and **filing level** (after section-weighted hard voting aggregates all "
        "chunks into one prediction per filing). Filing-level accuracy is the operationally "
        "meaningful number — it represents one prediction per real investment decision."
    )

    st.markdown("---")
    st.subheader("Why These Baselines?")
    st.info(
        "**Majority Class baseline** always predicts UP — achieving 61.1% accuracy purely "
        "from the 62% class imbalance in the 2023–2025 bull market test period. Its F1 Macro "
        "of 0.380 reveals it never correctly identifies a single DOWN case. This is why "
        "accuracy alone is a misleading metric here.\n\n"
        "**Logistic Regression baseline** uses hand-crafted numerical features (RSI, MACD, "
        "Volume, Fog Index, VADER sentiment) and achieves AUC 0.528. FinBERT T+20 reaches "
        "AUC 0.523 using only raw text — no engineered features. The ablation study "
        "(Advanced Analysis page) confirms NLP features add genuine value beyond technical indicators."
    )

    results = load_csv(os.path.join(RESULTS, "metrics", "master_results_table_final.csv"))
    if results is not None:
        st.subheader("Master Results Table")
        st.dataframe(
            results.style.highlight_max(
                subset=[c for c in ["F1_Macro","AUC","Accuracy"] if c in results.columns],
                color="#d4edda",
            ),
            use_container_width=True,
        )
        st.caption(
            "Filing-Level T+20 uses section-weighted hard voting to aggregate chunk predictions "
            "into one prediction per filing — this is the primary reported result. "
            "Hard voting is used rather than probability averaging because FinBERT fine-tuned "
            "on ~4,000 samples exhibits probability collapse (all probabilities cluster near 0.5). "
            "Hard voting only requires a chunk to be marginally above 0.5 to contribute a vote."
        )
    else:
        st.caption("Run 09_evaluate_models.py to generate the master results table.")

    st.markdown("---")
    img = load_image(os.path.join(RESULTS, "metrics", "results_comparison_chart.png"))
    if img:
        st.subheader("Performance Across Horizons")
        st.image(img, use_container_width=True)
        st.caption(
            "AUC improves monotonically from T+5 (0.498) through T+10 (0.515) to T+20 (0.523). "
            "T+5 being near-random is consistent with markets reacting to filing headlines immediately. "
            "The progressive improvement is the key evidence for gradual information diffusion."
        )

    st.markdown("---")
    st.subheader("Baseline Confusion Matrices")
    st.markdown(
        "These confusion matrices show what the baselines actually predict. "
        "The majority class matrix illustrates the UP-bias problem directly."
    )
    col1, col2 = st.columns(2)
    for fname, title in [
        ("baseline_majority_confusion.png", "Majority Class — always predicts UP"),
        ("baseline_lr_confusion.png",       "Logistic Regression — hand-crafted features"),
    ]:
        img = load_image(os.path.join(RESULTS, "baseline", fname))
        if img:
            (col1 if "majority" in fname else col2).image(img, caption=title, use_container_width=True)

    st.markdown("---")
    st.subheader("FinBERT Training Logs")
    st.markdown(
        "Training logs show validation F1 per epoch for each horizon model. "
        "Early stopping with patience=2 on validation F1 prevented overfitting. "
        "Three separate models were trained — one per prediction horizon."
    )
    for horizon in ["T5","T10","T20"]:
        log = load_csv(os.path.join(RESULTS,"metrics",f"finbert_training_log_{horizon}.csv"))
        if log is not None:
            with st.expander(f"FinBERT {horizon} — Training Log"):
                st.dataframe(log)
                st.caption(f"FinBERT {horizon}: AdamW optimiser, linear LR warmup, batch size 16, CPU only (Apple M2 MPS causes OOM at 8GB).")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("SHAP Feature Importance")
    st.markdown("---")

    st.markdown(
        "SHAP (SHapley Additive exPlanations — Lundberg & Lee, 2017) uses game theory to "
        "quantify each feature's contribution to individual predictions. These results come "
        "from the XGBoost hybrid ensemble trained on technical and readability features."
    )

    st.warning(
        "**Important context before reading these charts:** FinBERT confidence registers "
        "zero SHAP importance. This is a **pipeline design consequence**, not evidence that "
        "text adds no value. FinBERT inference was run on the test set only — it was never "
        "a training feature for XGBoost, so XGBoost has no learned relationship with it. "
        "The ablation study on the Advanced Analysis page provides the independent evidence "
        "for NLP contribution: NLP features alone achieve 54.9% vs technical-only 53.0%."
    )

    col1, col2 = st.columns(2)
    for fname, title, col in [
        ("shap_bar.png",      "Mean Absolute SHAP Value — overall feature ranking",   col1),
        ("shap_beeswarm.png", "Feature Impact Distribution — direction and magnitude", col2),
    ]:
        img = load_image(os.path.join(RESULTS, "shap", fname))
        if img:
            col.subheader(title)
            col.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("What the Model Learned")
    shap_df = pd.DataFrame({
        "Feature":     ["MACD","RSI","Volume Change","Flesch Reading Ease","Gunning Fog","VADER Sentiment","FinBERT Confidence"],
        "Mean |SHAP|": [0.446, 0.446, 0.319, 0.236, 0.211, 0.173, 0.000],
        "Interpretation": [
            "Price momentum at filing date — strongest signal; stocks already trending up are more likely to continue",
            "Relative Strength Index — momentum confirmation; equivalent importance to MACD",
            "Abnormal trading volume around filing release — institutional activity signal",
            "Easier-to-read filings associate with clearer outlook — Li (2008) effect confirmed",
            "Higher Fog Index (harder to read) associates with worse outcomes — managers obscure bad news",
            "Positive tone contributes but FinBERT provides deeper contextual understanding beyond raw sentiment",
            "Zero — pipeline constraint: FinBERT not run on training data. See ablation study.",
        ],
    })
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    st.info(
        "**Key takeaway:** Technical momentum features (MACD, RSI) dominate the XGBoost model "
        "because they capture the price context at the time of the filing. Readability features "
        "(Flesch, Fog) validate Li (2008)'s finding that filing complexity carries financial signal. "
        "The ablation study confirms NLP features add 1.9 percentage points beyond technical-only — "
        "independently of this SHAP analysis."
    )


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Backtest":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Strategic Backtest")
    st.markdown("---")

    st.markdown(
        "Four strategies were simulated on the 2023–2025 test set using pre-computed model "
        "predictions. All strategies use a 20-trading-day hold period matching the T+20 "
        "prediction horizon, £10,000 initial capital, 0.1% transaction cost per trade, "
        "and 20% maximum position size."
    )

    st.info(
        "**How to interpret these results:** All strategies underperform SPY in absolute return "
        "(+82.7%) over 2023–2025. This is **expected** under the Efficient Market Hypothesis — "
        "public information signals should not generate large abnormal returns after transaction "
        "costs. The relevant comparison is **risk-adjusted**: the Sortino ratio measures return "
        "per unit of downside risk. Strategies B and C achieve Sortino 4.30 versus SPY's ~1.2 — "
        "approximately 3.6× superior. At drawdown-equivalent leverage, Strategy C projects to "
        "~+219% versus SPY's +82.7%."
    )

    st.markdown("---")
    st.subheader("Strategy Definitions")
    st.markdown(
        "Each strategy applies a different filter to the model's UP predictions. "
        "Strategies A through C use 1× leverage. Strategy D is explicitly a leverage experiment "
        "applied to Strategy B's signals to isolate the leverage effect."
    )
    strat_def_df = pd.DataFrame({
        "Strategy":      ["A — All UP Signals","B — High Confidence","C — Sector Filtered","D — Leveraged (B × 2)"],
        "Signal Filter": [
            "All filings where model predicts UP",
            "UP predictions with confidence ≥ 0.72 (optimal threshold from sensitivity analysis)",
            "UP predictions in Financials + Technology sectors only (highest-accuracy sectors)",
            "Same signals as Strategy B (confidence ≥ 0.72) — leverage experiment",
        ],
        "Leverage":    ["1×","1×","1×","2×"],
        "Stop Loss":   ["None","None","None","−8%"],
        "Margin Cost": ["None","None","None","2% annualised"],
        "Trades":      ["309","124","89","124"],
    })
    st.dataframe(strat_def_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Headline Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strategy C — Max Drawdown",  "−3.7%",   "vs SPY −18.8%")
    col2.metric("Strategy B — Sortino",        "4.30",    "vs SPY ~1.2")
    col3.metric("Strategy C — Win Rate",       "71.9%",   "+21.9pp above chance")
    col4.metric("Strategy B — Total Return",   "+51.9%")
    col5.metric("SPY Benchmark — Sortino",     "~1.2",    "Strategies: up to 4.51")

    summary = load_csv(os.path.join(RESULTS, "backtest", "backtest_summary.csv"))
    if summary is not None:
        with st.expander("Full Strategy Summary CSV"):
            st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    st.subheader("Risk-Adjusted Comparison")
    comparison_df = pd.DataFrame({
        "Metric":                    ["Total return","Max drawdown","Sharpe","Sortino","Win rate","Trades"],
        "A — All UP":                ["+44.2%","−5.2%","1.08","2.03","63.1%","309"],
        "B — Conf ≥0.72":            ["+51.9%","−5.3%","1.68","4.30","67.7%","124"],
        "C — Fin+Tech":              ["+43.1%","−3.7%","1.87","4.30","71.9%","89"],
        "D — B × 2 Leverage":        ["+71.5%","−7.0%","1.51","4.51","65.3%","124"],
        "SPY Benchmark":             ["+82.7%","−18.8%","1.44","~1.2","—","—"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption(
        "Sortino ratio penalises only downside deviation — unlike Sharpe which penalises all "
        "volatility equally. For a long-only strategy, upside volatility is desirable. "
        "Strategy C Sortino of 4.30 vs SPY ~1.2 means 3.6× more return per unit of bad risk."
    )

    st.markdown("---")
    obs1, obs2 = st.columns(2)
    with obs1:
        st.markdown("**Strategy B vs D — Isolating the Leverage Effect**")
        st.markdown(
            "B and D trade the **identical signals** (confidence ≥ 0.72). "
            "D applies 2× leverage, −8% stop loss, and 2% annualised margin cost. "
            "Return: +51.9% → +71.5%. Drawdown: −5.3% → −7.0%. "
            "Sortino: 4.30 → 4.51. Leverage adds return more efficiently than "
            "it adds downside risk — the underlying signal handles leverage well."
        )
    with obs2:
        st.markdown("**Drawdown-Matched Signal Quality**")
        st.markdown(
            "Strategy C max drawdown −3.7% vs SPY −18.8%. If Strategy C were "
            "leveraged to match SPY's drawdown level (~5.1×), the projected return "
            "would be approximately +219% vs SPY's +82.7%. This illustrates the "
            "structural quality of the signal — not a trading recommendation."
        )

    st.markdown("---")
    for fname, title, explanation in [
        ("backtest_equity_curves.png",
         "Equity Curves — All Strategies vs SPY",
         "Strategies show substantially smoother growth curves with far smaller drawdowns than SPY. "
         "The key visual takeaway is not the ending value but the path — how calm versus volatile each line is."),
        ("backtest_by_ticker.png",
         "Performance by Ticker",
         "Some tickers contribute more to overall returns. Financials tickers tend to perform well "
         "consistent with the sector accuracy finding (73.6% Financials vs 40.0% Utilities)."),
        ("backtest_monthly.png",
         "Monthly Returns",
         "Month-by-month breakdown showing consistency of the signal. The 2024Q1 dip is visible — "
         "coinciding with peak Federal Reserve rate uncertainty."),
    ]:
        img = load_image(os.path.join(RESULTS, "backtest", fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.caption(explanation)
            st.markdown("---")

    trades = load_csv(os.path.join(RESULTS, "backtest", "backtest_trades.csv"))
    if trades is not None:
        with st.expander("View All Trades"):
            st.dataframe(trades, use_container_width=True)
            st.caption(f"{len(trades)} total trades across all strategies, 2023–2025.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Custom Backtest":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Custom Backtest Engine")
    st.markdown("---")

    st.markdown(
        "This engine replays the model's pre-computed predictions from `backtest_trades.csv` "
        "with configurable parameters. You can filter by ticker, sector, and date range, "
        "and adjust starting capital and transaction costs. "
        "Results update immediately after clicking **Run Backtest**."
    )
    st.info(
        "All results are derived from the same model predictions used in the main backtest — "
        "no additional model inference is required. This demonstrates that the pipeline "
        "produces practically usable outputs, not just research metrics."
    )

    with st.expander("Strategy Reference — what each strategy filters"):
        st.dataframe(pd.DataFrame({
            "Strategy":      ["A — All UP Signals","B — High Confidence","C — Sector Filtered","D — Leveraged (B × 2)"],
            "Signal Filter": [
                "All UP model predictions (no confidence filter)",
                "UP predictions with confidence ≥ 0.72 only",
                "UP predictions in Financials + Technology sectors only",
                "Same signals as B, with 2× leverage already applied in trade returns",
            ],
            "Leverage":  ["1×","1×","1×","2×"],
            "Stop Loss": ["None","None","None","−8%"],
            "Trades":    ["309","124","89","124"],
        }), use_container_width=True, hide_index=True)

    trades_path = os.path.join(RESULTS, "backtest", "backtest_trades.csv")
    trades_raw  = load_csv(trades_path)

    if trades_raw is None:
        st.error("backtest_trades.csv not found. Run 12_strategic_backtest.py first.")
        st.stop()

    trades_raw.columns = [c.strip().lower().replace(" ","_") for c in trades_raw.columns]

    col_map = {}
    for needed, candidates in {
        "ticker":    ["ticker","symbol","stock"],
        "entry_date":["entry_date","date","trade_date","filing_date"],
        "exit_date": ["exit_date"],
        "ret":       ["return","ret","trade_return","pnl_pct"],
        "allocated": ["allocated","position_size","pos_size"],
        "days_held": ["days_held","days","hold_days"],
        "strategy":  ["strategy","strat"],
    }.items():
        for c in candidates:
            if c in trades_raw.columns:
                col_map[needed] = c
                break

    if "ticker" in col_map:
        trades_raw["sector"] = trades_raw[col_map["ticker"]].map(
            lambda t: SECTOR_MAP.get(str(t).upper(),"Other")
        )

    for dc in ["entry_date","exit_date"]:
        if dc in col_map and col_map[dc] in trades_raw.columns:
            trades_raw[col_map[dc]] = pd.to_datetime(trades_raw[col_map[dc]], errors="coerce")

    date_col  = col_map.get("entry_date")
    exit_col  = col_map.get("exit_date")
    ret_col   = col_map.get("ret")
    alloc_col = col_map.get("allocated")
    strat_col = col_map.get("strategy")

    min_date    = trades_raw[date_col].min()  if date_col else None
    max_date    = trades_raw[date_col].max()  if date_col else None
    all_tickers = sorted(trades_raw[col_map["ticker"]].unique()) if "ticker" in col_map else []
    all_sectors = sorted(trades_raw["sector"].unique()) if "sector" in trades_raw.columns else []
    all_strats  = sorted(trades_raw[strat_col].unique()) if strat_col and strat_col in trades_raw.columns else []

    STRAT_DESC = {
        "Strategy A (all UP)":        "All UP predictions — no confidence filter (309 trades)",
        "Strategy B (conf>=0.72)":    "High confidence ≥ 0.72 — optimal threshold (124 trades)",
        "Strategy C (Fin+Tech only)": "Financials + Technology sectors only (89 trades)",
        "Strategy D (2x leveraged)":  "Strategy B signals + 2× leverage + −8% stop loss (124 trades)",
    }

    def get_desc(name):
        for k, v in STRAT_DESC.items():
            if k in name or name in k:
                return v
        return name

    st.markdown("---")
    st.subheader("Parameters")
    ctrl1, ctrl2 = st.columns(2)

    with ctrl1:
        st.markdown("**Universe Selection**")
        selection_mode = st.radio(
            "Select tickers by",
            ["All tickers","By sector","Manual selection"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if selection_mode == "All tickers":
            selected_tickers = all_tickers
            st.caption(f"All {len(all_tickers)} tickers selected across all sectors.")
        elif selection_mode == "By sector":
            selected_sectors = st.multiselect(
                "Sectors",
                all_sectors,
                default=all_sectors[:2] if len(all_sectors) >= 2 else all_sectors,
            )
            selected_tickers = [
                t for t in all_tickers
                if SECTOR_MAP.get(str(t).upper(),"Other") in selected_sectors
            ]
            st.caption(f"{len(selected_tickers)} tickers selected from: {', '.join(selected_sectors) if selected_sectors else 'none'}")
        else:
            selected_tickers = st.multiselect(
                "Individual tickers",
                all_tickers,
                default=all_tickers[:5] if len(all_tickers) >= 5 else all_tickers,
            )

        if all_strats:
            selected_strategy = st.selectbox(
                "Strategy",
                all_strats,
                index=0,
                help="A = all UP signals | B = high confidence ≥ 0.72 | C = Fin+Tech only | D = B signals with 2× leverage",
            )
            st.caption(get_desc(selected_strategy))
        else:
            selected_strategy = None

        if min_date and max_date:
            date_range = st.date_input(
                "Date range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
        else:
            date_range = None

    with ctrl2:
        st.markdown("**Capital & Risk Parameters**")
        starting_capital = st.number_input(
            "Starting capital (£)",
            min_value=1000, max_value=1_000_000, value=10_000, step=1000,
            help="Original backtest used £10,000.",
        )
        tx_cost = st.slider(
            "Transaction cost (%)", 0.0, 1.0, 0.1, 0.05,
            help="Broker cost per trade. Original: 0.1%. Realistic range: 0.05%–0.3%.",
        ) / 100
        st.caption(
            "Leverage and stop-loss are baked into Strategy D's trade returns already. "
            "Adjusting capital here scales all position sizes proportionally."
        )

    st.markdown("---")
    run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Configure parameters above then click **Run Backtest** to simulate.")
        st.stop()

    if not selected_strategy:
        st.warning("No strategy column found in backtest_trades.csv.")
        st.stop()

    df = trades_raw.copy()
    if strat_col and strat_col in df.columns:
        df = df[df[strat_col] == selected_strategy]
    if "ticker" in col_map and selected_tickers:
        df = df[df[col_map["ticker"]].isin(selected_tickers)]
    if date_col and date_range and len(date_range) == 2:
        df = df[
            (df[date_col] >= pd.Timestamp(date_range[0])) &
            (df[date_col] <= pd.Timestamp(date_range[1]))
        ]

    if len(df) == 0:
        st.warning("No trades match the selected filters. Try widening your selection.")
        st.stop()

    if not ret_col or ret_col not in df.columns:
        st.warning("Could not identify a return column in the trades file.")
        st.write("Columns found:", list(trades_raw.columns))
        st.stop()

    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce").fillna(0)
    if alloc_col and alloc_col in df.columns:
        df[alloc_col] = pd.to_numeric(df[alloc_col], errors="coerce").fillna(0)
    df = df.sort_values(date_col).reset_index(drop=True)

    original_cap = 10_000.0
    cap_scale    = starting_capital / original_cap
    eq_values = [starting_capital]
    eq_dates  = [df[date_col].iloc[0]]

    for _, row in df.iterrows():
        ret   = float(row[ret_col])
        alloc = float(row[alloc_col]) * cap_scale if alloc_col and alloc_col in df.columns else starting_capital * max_pos
        adj_ret = ret + 0.001 - tx_cost
        eq_values.append(eq_values[-1] + alloc * adj_ret)
        eq_dates.append(row[date_col])

    eq_s = pd.Series(eq_values, dtype=float)
    dt_s = pd.to_datetime(pd.Series(eq_dates, dtype=object), errors="coerce")

    r             = df[ret_col]
    final_capital = float(eq_s.iloc[-1])
    total_return  = (final_capital - starting_capital) / starting_capital * 100
    n_trades      = len(df)
    win_rate      = (r > 0).mean() * 100
    max_dd        = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min() * -1
    downside      = r[r < 0]
    down_std      = downside.std() if len(downside) > 1 else 1e-9
    sortino       = r.mean() / down_std * np.sqrt(252 / 20) if down_std > 0 else 0.0
    sharpe        = r.mean() / r.std()  * np.sqrt(252 / 20) if r.std()  > 0 else 0.0

    st.subheader(f"Results — {selected_strategy}")
    st.caption(get_desc(selected_strategy))

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Total Return",  f"{total_return:+.1f}%")
    c2.metric("Final Capital", f"£{final_capital:,.0f}")
    c3.metric("Trades",        str(n_trades))
    c4.metric("Win Rate",      f"{win_rate:.1f}%")
    c5.metric("Max Drawdown",  f"−{max_dd:.1f}%")
    c6.metric("Sortino",       f"{sortino:.2f}")
    c7.metric("Sharpe",        f"{sharpe:.2f}")

    st.markdown("---")
    st.subheader("Equity Curve")
    st.caption("Portfolio value over time. A smoother curve with fewer dips indicates better risk management.")
    eq_df = (
        pd.DataFrame({"Capital (£)": eq_s.values,"Date": dt_s.values})
        .dropna(subset=["Date"])
        .groupby("Date")["Capital (£)"].last()
        .reset_index().sort_values("Date").set_index("Date")
    )
    st.line_chart(eq_df, use_container_width=True)

    st.subheader("Drawdown")
    st.caption("Peak-to-trough loss at each point in time. Shallower = better capital preservation.")
    peak_eq = eq_df["Capital (£)"].cummax()
    dd_df   = ((eq_df["Capital (£)"] - peak_eq) / peak_eq * 100).rename("Drawdown (%)").to_frame()
    st.area_chart(dd_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Trade Log")
    st.caption(f"{n_trades} trades after filters. Win rate {win_rate:.1f}%.")
    show_cols = [c for c in [col_map.get("ticker"), date_col, exit_col, ret_col, alloc_col, col_map.get("days_held"), strat_col, "sector"] if c and c in df.columns]
    disp = df[show_cols].copy()
    if ret_col in disp.columns:
        disp[ret_col] = disp[ret_col].map(lambda x: f"{x:+.2%}")
    if alloc_col and alloc_col in disp.columns:
        disp[alloc_col] = disp[alloc_col].map(lambda x: f"£{x:,.0f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    if "ticker" in col_map:
        st.markdown("---")
        st.subheader("Per-Ticker Summary")
        st.caption("Breakdown by individual stock — useful for identifying which tickers drive performance.")
        tg = df.groupby(col_map["ticker"])[ret_col].agg(
            Trades="count",
            Win_Rate=lambda x: f"{(x>0).mean()*100:.0f}%",
            Avg_Return=lambda x: f"{x.mean()*100:+.2f}%",
            Total_Return=lambda x: f"{x.sum()*100:+.2f}%",
        ).reset_index()
        tg.columns = ["Ticker","Trades","Win Rate","Avg Return","Total Return"]
        st.dataframe(tg, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Error Analysis":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Error Analysis")
    st.markdown("---")

    st.markdown(
        "Understanding where and why the model fails is as important as knowing where it succeeds. "
        "This page breaks down prediction errors by sector, quarter, and individual filing "
        "to identify systematic failure patterns."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy",    "56.0%",  "Above 50% random chance")
    col2.metric("False Positive Rate", "27.7%",  "Predicted UP — stock fell")
    col3.metric("False Negative Rate", "16.3%",  "Predicted DOWN — stock rose")

    st.info(
        "**Why is false positive rate higher than false negative rate?** "
        "The model has an UP bias from the 62% class imbalance in training data. "
        "Even with inverse class weighting, the 2023–2025 test period was a bull market — "
        "the model more often predicts UP than DOWN, so it makes more UP errors."
    )

    st.markdown("---")
    for fname, title, explanation in [
        ("error_breakdown.png",
         "Error Breakdown & Sector Accuracy",
         "Financials achieves 73.6% — standardised regulatory language enables consistent patterns. "
         "Utilities at 40.0% is worst — small sample and utility-specific jargon not well represented in training. "
         "Technology at 51.1% suffers from frequent earnings surprises that dominate over filing language."),
        ("error_by_quarter.png",
         "Accuracy by Quarter (2023–2025)",
         "2024Q1 is the worst quarter at 37.5% accuracy — coinciding with peak Federal Reserve "
         "interest rate uncertainty. When macro factors dominate (rate decisions, Fed guidance), "
         "filing language loses predictive power. The model has no macroeconomic conditioning variables."),
        ("fp_return_distribution.png",
         "False Positive Return Distribution",
         "Distribution of actual returns on filings where the model predicted UP but the stock fell. "
         "Understanding the magnitude of false positive losses matters for risk management — "
         "are the losses small noise or large crashes?"),
    ]:
        img = load_image(os.path.join(RESULTS, "error_analysis", fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.caption(explanation)
            st.markdown("---")

    col1, col2 = st.columns(2)
    for fname, title, col in [
        ("worst_false_positives.csv", "Worst False Positives — predicted UP, stock fell most", col1),
        ("worst_false_negatives.csv", "Worst False Negatives — predicted DOWN, stock rose most", col2),
    ]:
        df_e = load_csv(os.path.join(RESULTS, "error_analysis", fname))
        if df_e is not None:
            col.subheader(title)
            col.dataframe(df_e, use_container_width=True)

    st.markdown("---")
    st.subheader("Systematic Failure Patterns")
    st.markdown(
        "These patterns reveal the model's structural limitations — "
        "not random errors but predictable failure modes with explanations:"
    )
    patterns_df = pd.DataFrame({
        "Pattern": [
            "Best sector: Financials (73.6%)",
            "Worst sector: Utilities (40.0%)",
            "Worst quarter: 2024Q1 (37.5%)",
            "ABBV consistently mispredicted",
            "AMD false negatives",
        ],
        "Root Cause": [
            "Highly standardised regulatory language — banks and insurers use consistent phrasing that the model learns reliably",
            "Small sample + utility-specific jargon not well represented in training data + macro sensitivity to rate changes",
            "Peak Federal Reserve rate-path uncertainty — macro signal overwhelms filing language in determining price direction",
            "Pharmaceutical pipeline binary outcomes (FDA approval/rejection) not captured in any filing language",
            "Rapid growth trajectory contradicts cautious forward-looking language — filing conservatism vs actual performance",
        ],
        "Implication": [
            "Sector selection (Strategy C) exploits this — restricting to Financials+Tech achieves 71.9% win rate",
            "Future work: add Longformer to capture more context; expand to more Utilities filings",
            "Future work: add macroeconomic conditioning variables (VIX, yield curve slope)",
            "Binary event risk is inherently unpredictable from text alone",
            "Semiconductor cycle dynamics require additional price momentum features",
        ],
    })
    st.dataframe(patterns_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Advanced Analysis":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Advanced Analysis")
    st.markdown("---")

    st.markdown(
        "This page contains nine analyses that together build a complete picture of the model's "
        "behaviour. They are ordered by importance to the research question — start with the "
        "CAAR event study and ROC curves, which are the two strongest pieces of evidence."
    )

    adv_path = os.path.join(RESULTS, "advanced")

    adv_sections = [
        ("B_caar_event_study.png",
         "1. CAAR Event Study — Independent Label Validation",
         "**This is the most important result in the project.** "
         "Cumulative Average Abnormal Return over [−10, +20] trading days around each filing date. "
         "UP-labelled filings outperform DOWN-labelled by +11.81 percentage points by day +20. "
         "Critically, this result is entirely independent of the model — labels are assigned from "
         "filing language alone. The progressive divergence building over 20 days (not jumping "
         "immediately on day zero) is direct empirical evidence for gradual information diffusion "
         "(Hong & Stein, 1999) and validates that the filing text labels have real economic meaning."),

        ("E_roc_curves.png",
         "2. ROC Curves — Monotonic AUC Improvement Across Horizons",
         "AUC improves monotonically: T+5 = 0.498 (random) → T+10 = 0.515 → T+20 = 0.523 (p < 0.001). "
         "The fact that AUC consistently improves with longer horizons — not randomly — is the "
         "statistical signature of gradual information diffusion. Each model was trained and "
         "evaluated independently, so the consistent pattern is not a coincidence."),

        ("H_ablation_study.png",
         "3. Ablation Study — Does Text Actually Help?",
         "NLP features only: 54.9%. Technical indicators only: 53.0%. All features: 57.7%. "
         "NLP features outperform technical-only by 1.9 percentage points. This directly "
         "answers the research question: SEC filing text adds genuine predictive value beyond "
         "price momentum signals alone. The full model combining both achieves the highest accuracy."),

        ("G_significance_tests.png",
         "4. Statistical Significance Tests",
         "Binomial test (win rate > 50%): p = 0.000032 — the 63.1% UP win rate is not random. "
         "Overall accuracy test: p = 0.0069. "
         "McNemar test vs majority baseline: p = 0.10 — not significant. "
         "The McNemar result is acknowledged as a limitation: both the ensemble and the majority "
         "baseline predict UP frequently in a bull market, so pairwise disagreement is low. "
         "The binomial and t-tests are the more relevant tests for the research question."),

        ("D_confidence_sensitivity.png",
         "5. Confidence Threshold Sensitivity",
         "How accuracy and trade volume vary as the confidence threshold increases. "
         "The optimal threshold of 0.72 (accuracy 66.9%, n=124 trades) is used in Strategy B. "
         "This shows that the model's confidence scores are calibrated and meaningful — "
         "higher confidence predictions are genuinely more accurate."),

        ("A_calibration_plot.png",
         "6. Calibration Plot (Reliability Diagram)",
         "ECE = 0.049 — the model is well calibrated. A perfectly calibrated model follows the "
         "diagonal: when it says 65% confident, it is correct approximately 65% of the time. "
         "Good calibration validates the confidence-filtered Strategy B — if probabilities were "
         "uncalibrated, filtering by confidence would not reliably select better predictions."),

        ("C_sector_breakdown.png",
         "7. Per-Sector Performance Breakdown",
         "Accuracy, precision, and recall across all 8 sectors. "
         "Financials 73.6% — highest accuracy. Utilities 40.0% — worst. "
         "The sector performance gap explains why Strategy C (Financials + Technology only) "
         "achieves higher win rates than Strategy A (all sectors)."),

        ("F_length_vs_accuracy.png",
         "8. Filing Length vs Prediction Accuracy",
         "Examines whether the number of chunks per filing (filing length) correlates with "
         "prediction accuracy. Longer filings produce more votes in the ensemble — "
         "does more text help or hurt?"),

        ("I_finbert_confusion_matrices.png",
         "9. FinBERT Confusion Matrices — All Horizons",
         "Per-class breakdown of predictions across T+5, T+10, and T+20 models. "
         "Shows how the model's ability to correctly identify DOWN cases improves "
         "with longer prediction horizons — consistent with the AUC trend."),
    ]

    for fname, title, explanation in adv_sections:
        img = load_image(os.path.join(adv_path, fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.caption(explanation)
            st.markdown("---")
        else:
            st.caption(f"{title} — run 14_advanced_analysis.py to generate this figure.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Prediction":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Live Prediction")
    st.markdown("---")

    st.markdown(
        "This interface runs the actual fine-tuned **FinBERT T+20 model** — the same model "
        "used in all evaluation and backtesting results — on any text you paste. "
        "Paste a paragraph from a real SEC 10-K or 10-Q filing and the model will predict "
        "whether the stock is likely to go **UP or DOWN** in the 20 trading days "
        "(approximately 1 month) following the filing date."
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filing-level accuracy", "60.9%", "430 test filings 2023–2025")
    col_b.metric("UP win rate",           "63.1%", "p < 0.001, binomial test")
    col_c.metric("Confidence threshold",  "0.72",  "66.9% accuracy when used")

    st.info(
        "**How the prediction works:** The pasted text is tokenised and split into 512-token "
        "chunks (BERT's architectural limit). Each chunk receives a UP probability from FinBERT. "
        "Simple majority voting (without section weighting, since section identity is unknown "
        "for arbitrary text) produces the final prediction. The confidence score is the average "
        "UP probability across all chunks. Predictions are based on filing language only — "
        "macroeconomic conditions are not modelled."
    )

    @st.cache_resource(show_spinner="Loading FinBERT T+20 model (first load only)...")
    def load_finbert_model():
        import torch
        from transformers import BertForSequenceClassification, AutoTokenizer
        model_dir = os.path.join(PROJECT_ROOT, "03_Models", "finbert_champion_T20")
        if not os.path.exists(model_dir):
            return None, None
        tok   = AutoTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
        model.eval()
        return tok, model

    def chunk_text(text, tokenizer, max_len=512, stride=50):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks, start = [], 0
        while start < len(tokens):
            end = min(start + max_len - 2, len(tokens))
            chunks.append(tokens[start:end])
            if end == len(tokens):
                break
            start += max_len - stride
        return chunks

    def predict_text(text, tokenizer, model):
        import torch
        token_chunks = chunk_text(text, tokenizer)
        if not token_chunks:
            return None
        results = []
        with torch.no_grad():
            for tc in token_chunks:
                ids  = [tokenizer.cls_token_id] + tc + [tokenizer.sep_token_id]
                ids  = ids[:512]
                pad  = [tokenizer.pad_token_id] * (512 - len(ids))
                iids = torch.tensor([ids + pad])
                mask = torch.tensor([[1]*len(ids) + [0]*len(pad)])
                out  = model(input_ids=iids, attention_mask=mask)
                prob_up = float(torch.softmax(out.logits, dim=1)[0][1])
                results.append({"n_tokens": len(tc), "prob_up": round(prob_up,4), "pred": int(prob_up >= 0.5)})
        n_up  = sum(r["pred"] for r in results)
        conf  = sum(r["prob_up"] for r in results) / len(results)
        return {"prediction": int(n_up >= len(results) - n_up), "confidence": conf,
                "n_chunks": len(results), "n_up_votes": n_up,
                "n_down_votes": len(results) - n_up, "chunk_details": results}

    EXAMPLE_TEXT = (
        "Risk Factors and Forward-Looking Statements. Our revenues depend significantly on "
        "general economic conditions and the level of consumer spending. Demand for our products "
        "has increased substantially over the prior year period, driven by strong performance in "
        "our cloud services division. We expect continued growth in recurring revenue streams and "
        "have secured several multi-year enterprise contracts. Management believes our diversified "
        "portfolio positions us well to capitalise on market opportunities while managing exposure "
        "to adverse macroeconomic developments. Our strong balance sheet and cash generation "
        "capabilities provide us with the financial flexibility to invest in growth initiatives "
        "and return capital to shareholders."
    )

    col_input, col_btn = st.columns([4,1])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load example text", use_container_width=True):
            st.session_state["live_text"] = EXAMPLE_TEXT
            st.caption("Example loaded — click Run Prediction")

    text_input = st.text_area(
        "Paste SEC filing text (MD&A, Risk Factors, etc.):",
        value=st.session_state.get("live_text",""),
        height=250,
        key="live_text_area",
        placeholder="Paste a paragraph from Item 7 (MD&A) or Item 1A (Risk Factors)...",
    )

    if st.button("Run T+20 Prediction", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please paste some filing text first.")
        else:
            with st.spinner("Loading FinBERT T+20 model..."):
                tok, mdl = load_finbert_model()
            if tok is None:
                st.error(
                    "FinBERT T+20 model not found at 03_Models/finbert_champion_T20/. "
                    "Run: python 02_Code/models/08_train_finbert.py --horizon T20"
                )
            else:
                with st.spinner("Tokenising and running inference across chunks..."):
                    result = predict_text(text_input, tok, mdl)
                st.markdown("---")
                direction = "UP" if result["prediction"] == 1 else "DOWN"
                if result["prediction"] == 1:
                    st.success(
                        f"Predicted direction: **{direction}** — "
                        f"model expects positive price movement over the next 20 trading days"
                    )
                else:
                    st.error(
                        f"Predicted direction: **{direction}** — "
                        f"model expects negative price movement over the next 20 trading days"
                    )

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Prediction",     direction)
                c2.metric("Avg Confidence", f"{result['confidence']:.1%}")
                c3.metric("Chunks processed", result["n_chunks"])
                c4.metric("Vote split",     f"{result['n_up_votes']} UP / {result['n_down_votes']} DOWN")

                st.markdown("---")
                st.markdown("**Chunk-level breakdown** — how each 512-token window voted:")
                cdf = pd.DataFrame(result["chunk_details"])
                cdf.index      = [f"Chunk {i+1}" for i in range(len(cdf))]
                cdf["vote"]    = cdf["pred"].map({1:"UP","0":"DOWN",0:"DOWN"})
                cdf["prob_up"] = cdf["prob_up"].map(lambda x: f"{x:.1%}")
                cdf = cdf.rename(columns={"n_tokens":"Tokens","prob_up":"P(UP)","vote":"Vote"})
                st.dataframe(cdf[["Tokens","P(UP)","Vote"]], use_container_width=True)

                conf = result["confidence"]
                strength = "high" if abs(conf-0.5)>0.15 else "moderate" if abs(conf-0.5)>0.07 else "low"
                st.caption(
                    f"Average confidence: {conf:.1%} ({strength} conviction). "
                    f"The model achieves 66.9% accuracy when confidence ≥ 0.72 on the held-out test set. "
                    "Note: probability collapse — FinBERT trained on ~4,000 samples produces "
                    "probabilities clustered near 0.5. The chunk count and vote split are "
                    "more informative than the raw probability value."
                )

    st.markdown("---")
    st.caption(
        "Model: ProsusAI/finbert fine-tuned on 4,000 SEC 10-K/10-Q training chunks, T+20 label. "
        "Layer-wise freezing: layers 0–9 frozen (87% of parameters), layers 10–11 + classifier trained. "
        "Trained on filings pre-2021 | Validated 2021–2022 | Tested 2023–2025."
    )