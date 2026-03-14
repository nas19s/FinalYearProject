"""
app.py — StockDrift Research Dashboard
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
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Georgia', 'Times New Roman', serif;
    }
    h1 { font-size: 1.9rem !important; font-weight: 700; letter-spacing: -0.02em; }
    h2 { font-size: 1.3rem !important; font-weight: 600; color: #1a1a2e; }
    h3 { font-size: 1.1rem !important; font-weight: 600; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    section[data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem;
        padding: 4px 0;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #666 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-family: 'Courier New', monospace;
        color: #1a1a2e !important;
    }

    /* Dividers */
    hr { border: none; border-top: 1px solid #e8e8e8; margin: 1.5rem 0; }

    /* Info/warning boxes */
    div[data-testid="stInfo"] {
        background-color: #f0f4ff;
        border-left: 3px solid #3a5bd9;
        border-radius: 0 4px 4px 0;
        font-size: 0.9rem;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }

    /* Caption text */
    div[data-testid="stCaptionContainer"] {
        color: #888;
        font-size: 0.8rem;
        font-style: italic;
    }

    /* Hide Streamlit branding */
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


st.sidebar.markdown("## StockDrift")
st.sidebar.markdown("SEC Filing → Price Direction")
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "Overview",
    "Data & EDA",
    "Model Results",
    "SHAP Explainability",
    "Live Prediction",
    "Backtest",
    "Error Analysis",
    "Advanced Analysis",
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Final Year CS Project<br>University of Birmingham<br>2025–2026</small>",
    unsafe_allow_html=True,
)



# PAGE: OVERVIEW

if page == "Overview":
    st.title("StockDrift")
    st.markdown(
        "**Predicting Stock Price Direction from SEC Regulatory Filings using FinBERT**"
    )
    st.markdown("---")

    st.markdown("""
    > **Research Question:** Can the textual content of SEC 10-K and 10-Q regulatory filings
    > predict the direction of stock price movement in the weeks following their release?
    """)

    st.markdown(
        "This project tests the semi-strong form of the Efficient Market Hypothesis (Fama, 1970). "
        "Under EMH, publicly available information should be immediately reflected in prices, "
        "leaving no exploitable signal. The hypothesis tested here is that complex regulatory "
        "documents are not immediately and fully processed by the market — and that "
        "language-model-derived signals may capture residual predictive content."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filings collected", "1,823")
    col2.metric("S&P 500 tickers", "50")
    col3.metric("Text chunks", "105,322")
    col4.metric("Years of data", "2016–2026")

    st.markdown("---")
    st.subheader("Key Results")

    results = load_csv(os.path.join(RESULTS, "metrics", "master_results_table_final.csv"))
    if results is not None:
        st.dataframe(
            results.style.highlight_max(
                subset=[c for c in ["F1_Macro", "AUC", "Accuracy"] if c in results.columns],
                color="#d4edda",
            ),
            use_container_width=True,
        )
    else:
        st.caption("Run 09_evaluate_models.py to generate the master results table.")

    st.markdown("---")
    st.subheader("Core Finding — Prediction Horizon")

    col1, col2, col3 = st.columns(3)
    col1.metric("T+5 AUC  (1 week)",   "0.498", "≈ random")
    col2.metric("T+10 AUC (2 weeks)",  "0.515", "+0.017")
    col3.metric("T+20 AUC (1 month)",  "0.523", "+0.025")

    st.info(
        "T+5 prediction is near-random, consistent with rapid market reactions to new filings. "
        "The T+20 result (AUC = 0.523, p < 0.001) suggests that the full informational content "
        "of complex regulatory language takes approximately one month to be reflected in prices — "
        "consistent with Hong & Stein's (1999) gradual information diffusion hypothesis."
    )

    st.markdown("---")
    st.subheader("Pipeline Summary")
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
        Backtest + SHAP Analysis
    """, language="text")



# PAGE: DATA & EDA

    st.title("Data & Exploratory Analysis")
    st.markdown("---")

    eda_path = os.path.join(RESULTS, "eda")
    figures = {
        "Label Distributions (T+5, T+10, T+20)": "fig1_label_distributions.png",
        "Filing Timeline by Year":               "fig2_filing_timeline.png",
        "Chunk Token Length Distribution":       "fig3_chunk_lengths.png",
        "Section Coverage per Filing":           "fig4_section_coverage.png",
        "Labels by Form Type (10-K vs 10-Q)":   "fig5_labels_by_form_type.png",
        "Top 20 Tickers by Filing Count":        "fig6_top_tickers.png",
    }

    for title, fname in figures.items():
        img = load_image(os.path.join(eda_path, fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.markdown("---")

    st.subheader("Dataset Statistics")
    meta_path = os.path.join(DATA_DIR, "sec_metadata.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        col1, col2 = st.columns(2)
        col1.metric("Total filings",    f"{len(meta):,}")
        col1.metric("10-K annual",      f"{(meta['form_type']=='10-K').sum():,}")
        col2.metric("Unique tickers",   f"{meta['ticker'].nunique():,}")
        col2.metric("10-Q quarterly",   f"{(meta['form_type']=='10-Q').sum():,}")
    else:
        st.caption("Run 02_get_sec_data.py to populate sec_metadata.csv.")

    st.markdown("---")
    st.subheader("Section Extraction Weights")
    st.markdown(
        "Four sections were extracted from each filing. Section weights reflect their "
        "expected informational value, based on prior literature (Li, 2008; Loughran & McDonald, 2011):"
    )
    weight_df = pd.DataFrame({
        "Section":     ["Item 1A — Risk Factors", "Item 7 — MD&A",
                         "Item 7A — Market Risk",  "Item 9A — Internal Controls"],
        "Weight":      [1.0, 1.0, 0.6, 0.8],
        "Rationale":   [
            "Forward-looking risk disclosure; legally required",
            "Management's discussion of results; most studied in literature",
            "Quantitative market risk disclosures; less narrative content",
            "Internal control quality; governance signal",
        ],
    })
    st.dataframe(weight_df, use_container_width=True, hide_index=True)



# PAGE: MODEL RESULTS

elif page == "Model Results":
    st.title("Model Results")
    st.markdown("---")

    results = load_csv(os.path.join(RESULTS, "metrics", "master_results_table_final.csv"))
    if results is not None:
        st.subheader("Master Results Table")
        st.dataframe(results, use_container_width=True)
    else:
        st.caption("Run 09_evaluate_models.py to generate results.")

    st.markdown("---")
    img = load_image(os.path.join(RESULTS, "metrics", "results_comparison_chart.png"))
    if img:
        st.subheader("Performance Across Horizons")
        st.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("Baseline Comparison")
    st.markdown(
        "Two baselines are used. The majority class baseline always predicts UP — "
        "achieving 61.1% accuracy purely from class imbalance, but with F1 macro of 0.380 "
        "since it never correctly identifies DOWN cases. "
        "The logistic regression baseline uses hand-crafted numerical features (RSI, MACD, "
        "readability scores) and achieves AUC 0.528. The FinBERT T+20 model reaches "
        "comparable AUC (0.523) using only raw text with no engineered features."
    )

    col1, col2 = st.columns(2)
    for fname, title in [
        ("baseline_majority_confusion.png", "Majority Class Baseline"),
        ("baseline_lr_confusion.png",       "Logistic Regression Baseline"),
    ]:
        img = load_image(os.path.join(RESULTS, "baseline", fname))
        if img:
            (col1 if "majority" in fname else col2).image(
                img, caption=title, use_container_width=True
            )

    st.markdown("---")
    st.subheader("FinBERT Training Logs")
    for horizon in ["T5", "T10", "T20"]:
        log = load_csv(os.path.join(
            RESULTS, "metrics", f"finbert_training_log_{horizon}.csv"
        ))
        if log is not None:
            with st.expander(f"FinBERT {horizon} — Training Log"):
                st.dataframe(log)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: SHAP
# ──────────────────────────────────────────────────────────────────────────────
elif page == "SHAP Explainability":
    st.title("SHAP Feature Importance")
    st.markdown("---")

    st.markdown(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's contribution "
        "to individual predictions. The analysis is run on the XGBoost hybrid ensemble "
        "trained on technical and readability features."
    )

    col1, col2 = st.columns(2)
    for fname, title, col in [
        ("shap_bar.png",      "Mean Absolute SHAP Value",    col1),
        ("shap_beeswarm.png", "Feature Impact Distribution", col2),
    ]:
        img = load_image(os.path.join(RESULTS, "shap", fname))
        if img:
            col.subheader(title)
            col.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("Findings")

    shap_df = pd.DataFrame({
        "Feature":            ["MACD", "RSI", "Volume Change", "Flesch Reading Ease",
                               "Gunning Fog", "VADER Sentiment", "FinBERT Confidence"],
        "Mean |SHAP|":        [0.446, 0.446, 0.319, 0.236, 0.211, 0.173, 0.000],
        "Notes":              [
            "Strongest predictor — momentum context at filing date",
            "Momentum indicator — equivalent importance to MACD",
            "Abnormal trading volume around filing release",
            "Readability predicts returns (Li, 2008)",
            "Filing complexity carries independent signal",
            "Tone contributes; FinBERT provides deeper contextual understanding",
            "Zero — FinBERT not inferred on training data; see ablation study",
        ],
    })
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    st.info(
        "FinBERT confidence registers zero SHAP importance because FinBERT inference was "
        "run on the test set only and could not be included as a training feature for the "
        "XGBoost ensemble. The ablation study (Advanced Analysis page) provides independent "
        "evidence that NLP features contribute beyond technical indicators."
    )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Backtest":
    st.title("Strategic Backtest")
    st.markdown("---")

    st.markdown(
        "Four strategies were simulated on the 2023–2025 test set. "
        "All strategies use a 20-trading-day hold period (matching the T+20 prediction horizon), "
        "£10,000 initial capital, 0.1% transaction cost per trade, and a 20% maximum position size."
    )

    summary = load_csv(os.path.join(RESULTS, "backtest", "backtest_summary.csv"))
    if summary is not None:
        st.subheader("Strategy Summary")
        st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy C — Max Drawdown", "−3.7%",  "vs SPY −18.8%")
    col2.metric("Strategy B — Sortino",       "4.30",   "vs SPY ~1.2")
    col3.metric("Strategy C — Win Rate",      "71.9%")
    col4.metric("Strategy B — Total Return",  "+51.9%")

    st.markdown("---")

    st.subheader("Risk-Adjusted Performance")
    comparison_df = pd.DataFrame({
        "Metric":         ["Total return", "Max drawdown", "Sharpe ratio",
                           "Sortino ratio", "Win rate", "Trades"],
        "Strategy A":     ["+44.2%", "−5.2%", "1.08", "2.03", "63.1%", "309"],
        "Strategy B":     ["+51.9%", "−5.3%", "1.68", "4.30", "67.7%", "124"],
        "Strategy C":     ["+43.1%", "−3.7%", "1.87", "4.30", "71.9%", "89"],
        "Strategy D":     ["+71.5%", "−7.0%", "1.51", "4.51", "65.3%", "124"],
        "SPY benchmark":  ["+82.7%", "−18.8%", "1.44", "~1.2",  "—",   "—"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.info(
        "All strategies underperform SPY in absolute return over the 2023–2025 test period, "
        "which is consistent with the semi-strong form of the Efficient Market Hypothesis — "
        "public information signals are not expected to generate large abnormal returns after "
        "transaction costs. The Sortino ratio (which penalises only downside deviation) "
        "is the more relevant measure for this type of long-only strategy: Strategy B's "
        "Sortino of 4.30 versus SPY's estimated 1.2 indicates substantially more efficient "
        "use of downside risk budget. Strategy C's maximum drawdown of −3.7% compares "
        "favourably against SPY's −18.8% over the same period."
    )

    for fname, title in [
        ("backtest_equity_curves.png", "Equity Curves — All Strategies vs SPY"),
        ("backtest_by_ticker.png",     "Performance by Ticker"),
        ("backtest_monthly.png",       "Monthly Returns"),
    ]:
        img = load_image(os.path.join(RESULTS, "backtest", fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.markdown("---")

    trades = load_csv(os.path.join(RESULTS, "backtest", "backtest_trades.csv"))
    if trades is not None:
        with st.expander("View All Trades"):
            st.dataframe(trades, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: ERROR ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Error Analysis":
    st.title("Error Analysis")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy",    "56.0%")
    col2.metric("False Positive Rate", "27.7%",  "Predicted UP — stock fell")
    col3.metric("False Negative Rate", "16.3%",  "Predicted DOWN — stock rose")

    st.markdown("---")

    for fname, title, caption in [
        ("error_breakdown.png",
         "Error Breakdown & Sector Accuracy",
         "Accuracy, false positive, and false negative rates by sector."),
        ("error_by_quarter.png",
         "Accuracy by Quarter",
         "Temporal breakdown of prediction accuracy across 2023–2025."),
        ("fp_return_distribution.png",
         "False Positive Return Distribution",
         "Distribution of actual returns on filings where the model predicted UP but the stock fell."),
    ]:
        img = load_image(os.path.join(RESULTS, "error_analysis", fname))
        if img:
            st.subheader(title)
            st.caption(caption)
            st.image(img, use_container_width=True)
            st.markdown("---")

    col1, col2 = st.columns(2)
    for fname, title, col in [
        ("worst_false_positives.csv", "Worst False Positives", col1),
        ("worst_false_negatives.csv", "Worst False Negatives", col2),
    ]:
        df = load_csv(os.path.join(RESULTS, "error_analysis", fname))
        if df is not None:
            col.subheader(title)
            col.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Failure Pattern Summary")

    patterns_df = pd.DataFrame({
        "Pattern":     [
            "Best sector: Financials (73.6%)",
            "Worst sector: Utilities (40.0%)",
            "Worst quarter: 2024Q1 (37.5%)",
            "ABBV consistently mispredicted",
            "AMD false negatives",
        ],
        "Explanation": [
            "Standardised regulatory language; model learns consistent patterns",
            "Small sample size and utility-specific terminology not well represented in training",
            "Federal Reserve rate-path uncertainty dominated returns; macro signal overwhelms text",
            "Pharmaceutical pipeline binary outcomes not captured in filing language",
            "Rapid growth trajectory contradicts cautious forward-looking language in filings",
        ],
    })
    st.dataframe(patterns_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: ADVANCED ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Advanced Analysis":
    st.title("Advanced Analysis")
    st.markdown("---")

    adv_path = os.path.join(RESULTS, "advanced")

    sections = [
        ("A_calibration_plot.png",
         "A. Calibration (Reliability Diagram)",
         "Plots predicted probability against observed accuracy. "
         "A perfectly calibrated model follows the diagonal. "
         "ECE = 0.049 indicates the model is well calibrated."),

        ("B_caar_event_study.png",
         "B. CAAR Event Study",
         "Cumulative Average Abnormal Return in the window [−10, +20] trading days "
         "around each filing date. UP-labelled filings outperform DOWN-labelled by "
         "+11.81 percentage points by day +20, independently validating label quality."),

        ("C_sector_breakdown.png",
         "C. Per-Sector Performance",
         "Accuracy, precision, and recall across all 8 sectors. "
         "Financials achieve 73.6%; Utilities 40.0%."),

        ("D_confidence_sensitivity.png",
         "D. Confidence Threshold Sensitivity",
         "Accuracy and trade count as a function of the confidence threshold. "
         "The optimal threshold of 0.72 is used in Strategy B."),

        ("E_roc_curves.png",
         "E. ROC Curves — T+5, T+10, T+20",
         "AUC increases monotonically across horizons: 0.498 → 0.515 → 0.523. "
         "Supports the gradual information diffusion interpretation."),

        ("F_length_vs_accuracy.png",
         "F. Filing Length vs Prediction Accuracy",
         "Examines whether filing length (number of chunks) correlates with accuracy."),

        ("G_significance_tests.png",
         "G. Statistical Significance Tests",
         "Binomial test (win rate > 50%): p = 0.000032. "
         "Overall accuracy test: p = 0.0069. "
         "McNemar test vs majority baseline: p = 0.10 (not significant — acknowledged)."),

        ("H_ablation_study.png",
         "H. Ablation Study",
         "NLP features only: 54.9%. Technical only: 53.0%. All features: 57.7%. "
         "Confirms that text-derived features add predictive value beyond momentum indicators."),

        ("I_finbert_confusion_matrices.png",
         "I. FinBERT Confusion Matrices — All Horizons",
         "Per-class breakdown across T+5, T+10, and T+20 models."),
    ]

    for fname, title, caption in sections:
        img = load_image(os.path.join(adv_path, fname))
        if img:
            st.subheader(title)
            st.caption(caption)
            st.image(img, use_container_width=True)
            st.markdown("---")
        else:
            st.caption(f"{title} — run 14_advanced_analysis.py to generate.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Live Prediction":
    st.title("Live Prediction")
    st.markdown("---")

    st.markdown(
        "Paste text from a real SEC 10-K or 10-Q filing below. "
        "The fine-tuned **FinBERT T+20 model** will predict whether the stock is likely "
        "to go UP or DOWN in the 20 trading days (~1 month) following the filing date. "
        "This runs the same model used in all evaluation and backtesting results."
    )

    st.info(
        "The model achieves 60.9% filing-level accuracy and a 63.1% UP win rate on the "
        "2023–2025 test set. Predictions are based on filing language only — "
        "macroeconomic conditions are not considered."
    )

    # ── model loader ──────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading FinBERT T+20 model...")
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
        chunks = []
        start  = 0
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
        chunk_results = []
        with torch.no_grad():
            for chunk_tokens in token_chunks:
                ids = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
                ids = ids[:512]
                padding    = [tokenizer.pad_token_id] * (512 - len(ids))
                input_ids  = torch.tensor([ids + padding])
                attn_mask  = torch.tensor([[1] * len(ids) + [0] * len(padding)])
                out        = model(input_ids=input_ids, attention_mask=attn_mask)
                probs      = torch.softmax(out.logits, dim=1)[0]
                prob_up    = float(probs[1])
                chunk_results.append({
                    "n_tokens": len(chunk_tokens),
                    "prob_up":  round(prob_up, 4),
                    "pred":     1 if prob_up >= 0.5 else 0,
                })
        n_up  = sum(r["pred"] for r in chunk_results)
        n_down = len(chunk_results) - n_up
        conf  = sum(r["prob_up"] for r in chunk_results) / len(chunk_results)
        return {
            "prediction":    1 if n_up >= n_down else 0,
            "confidence":    conf,
            "n_chunks":      len(chunk_results),
            "n_up_votes":    n_up,
            "n_down_votes":  n_down,
            "chunk_details": chunk_results,
        }

    # ── example text ──────────────────────────────────────────────────────────
    EXAMPLE_TEXT = (
        "Risk Factors and Forward-Looking Statements. "
        "Our revenues depend significantly on general economic conditions and the level of "
        "consumer spending. Demand for our products has increased substantially over the prior "
        "year period, driven by strong performance in our cloud services division. We expect "
        "continued growth in recurring revenue streams and have secured several multi-year "
        "enterprise contracts. Management believes our diversified portfolio positions us well "
        "to capitalise on market opportunities while managing exposure to adverse macroeconomic "
        "developments. Our strong balance sheet and cash generation capabilities provide us with "
        "the financial flexibility to invest in growth initiatives and return capital to shareholders."
    )

    col_input, col_btn = st.columns([4, 1])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load example text", use_container_width=True):
            st.session_state["live_text"] = EXAMPLE_TEXT

    text_input = st.text_area(
        "Paste SEC filing text (MD&A, Risk Factors, etc.):",
        value=st.session_state.get("live_text", ""),
        height=250,
        key="live_text_area",
        placeholder="e.g. paste a paragraph from Item 7 (Management Discussion and Analysis)...",
    )

    predict_btn = st.button("Run T+20 Prediction", type="primary", use_container_width=True)

    if predict_btn:
        if not text_input.strip():
            st.warning("Please paste some filing text first.")
        else:
            with st.spinner("Running FinBERT T+20 inference..."):
                tok, mdl = load_finbert_model()

            if tok is None:
                st.error(
                    "FinBERT T+20 model not found at 03_Models/finbert_champion_T20/. "
                    "Run 08_train_finbert.py --horizon T20 first."
                )
            else:
                with st.spinner("Analysing chunks..."):
                    result = predict_text(text_input, tok, mdl)

                st.markdown("---")

                direction = "UP" if result["prediction"] == 1 else "DOWN"
                if result["prediction"] == 1:
                    st.success(f"Predicted direction: **{direction}** — positive price movement expected over T+20")
                else:
                    st.error(f"Predicted direction: **{direction}** — negative price movement expected over T+20")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction",      direction)
                c2.metric("Avg Confidence",  f"{result['confidence']:.1%}")
                c3.metric("Text Chunks",     result["n_chunks"])
                c4.metric("Vote Split",
                          f"{result['n_up_votes']} UP / {result['n_down_votes']} DOWN")

                st.markdown("**Chunk-level detail:**")
                chunk_df = pd.DataFrame(result["chunk_details"])
                chunk_df.index = [f"Chunk {i+1}" for i in range(len(chunk_df))]
                chunk_df["vote"]    = chunk_df["pred"].map({1: "UP", 0: "DOWN"})
                chunk_df["prob_up"] = chunk_df["prob_up"].map(lambda x: f"{x:.1%}")
                chunk_df = chunk_df.rename(columns={
                    "n_tokens": "Tokens", "prob_up": "P(UP)", "vote": "Vote"
                })
                st.dataframe(chunk_df[["Tokens", "P(UP)", "Vote"]], use_container_width=True)

                conf = result["confidence"]
                strength = (
                    "high"     if abs(conf - 0.5) > 0.15 else
                    "moderate" if abs(conf - 0.5) > 0.07 else
                    "low"
                )
                st.caption(
                    f"Average chunk confidence: {conf:.1%} ({strength} conviction). "
                    "Prediction based on filing language only — macroeconomic factors are not modelled."
                )

    st.markdown("---")
    st.caption(
        "Model: ProsusAI/finbert fine-tuned on SEC 10-K/10-Q filings, T+20 label. "
        "Trained on filings pre-2021, validated 2021–2022, tested 2023+."
    )