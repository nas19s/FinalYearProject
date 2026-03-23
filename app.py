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

# FIX: label_visibility="collapsed" prevents the empty-label warning crash
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
st.sidebar.markdown(
    "<small>Final Year CS Project<br>University of Birmingham<br>2025–2026</small>",
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
    st.markdown("---")
    st.markdown("""
    > **Research Question:** Can the textual content of SEC 10-K and 10-Q regulatory filings
    > predict the direction of stock price movement in the weeks following their release?
    """)
    st.markdown(
        "This project tests the semi-strong form of the Efficient Market Hypothesis (Fama, 1970). "
        "Under EMH, publicly available information should be immediately reflected in prices. "
        "The hypothesis here is that complex regulatory documents are not immediately and fully "
        "processed by the market — and that language-model-derived signals may capture residual "
        "predictive content."
    )
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filings collected", "1,823")
    col2.metric("S&P 500 tickers",   "50")
    col3.metric("Text chunks",        "105,322")
    col4.metric("Years of data",      "2016–2026")
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
    else:
        st.caption("Run 09_evaluate_models.py to generate the master results table.")
    st.markdown("---")
    st.subheader("Core Finding — Prediction Horizon")
    col1, col2, col3 = st.columns(3)
    col1.metric("T+5 AUC  (1 week)",  "0.498", "≈ random")
    col2.metric("T+10 AUC (2 weeks)", "0.515", "+0.017")
    col3.metric("T+20 AUC (1 month)", "0.523", "+0.025")
    st.info(
        "T+5 prediction is near-random, consistent with rapid market reactions to new filings. "
        "The T+20 result (AUC = 0.523, p < 0.001) suggests the full informational content of "
        "complex regulatory language takes approximately one month to be reflected in prices — "
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


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data & EDA":
# ══════════════════════════════════════════════════════════════════════════════
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
    any_shown = False
    for title, fname in figures.items():
        img = load_image(os.path.join(eda_path, fname))
        if img:
            st.subheader(title)
            st.image(img, use_container_width=True)
            st.markdown("---")
            any_shown = True
    if not any_shown:
        st.info("No EDA figures found. Run 06_eda_analysis.py to generate them.")
    st.subheader("Dataset Statistics")
    meta_path = os.path.join(DATA_DIR, "sec_metadata.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        col1, col2 = st.columns(2)
        col1.metric("Total filings",  f"{len(meta):,}")
        col1.metric("10-K annual",    f"{(meta['form_type']=='10-K').sum():,}")
        col2.metric("Unique tickers", f"{meta['ticker'].nunique():,}")
        col2.metric("10-Q quarterly", f"{(meta['form_type']=='10-Q').sum():,}")
    else:
        st.caption("Run 02_get_sec_data.py to populate sec_metadata.csv.")
    st.markdown("---")
    st.subheader("Section Extraction Weights")
    st.markdown("Four sections were extracted per filing. Weights are grounded in Li (2008) and Loughran & McDonald (2011).")
    weight_df = pd.DataFrame({
        "Section":   ["Item 1A — Risk Factors","Item 7 — MD&A","Item 7A — Market Risk","Item 9A — Internal Controls"],
        "Weight":    [1.0, 1.0, 0.6, 0.8],
        "Rationale": [
            "Forward-looking risk disclosure; legally required",
            "Management's discussion of results; most studied in literature",
            "Quantitative market risk disclosures; less narrative content",
            "Internal control quality; governance signal",
        ],
    })
    st.dataframe(weight_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
# ══════════════════════════════════════════════════════════════════════════════
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
        "61.1% accuracy from class imbalance, but F1 macro 0.380 since it never identifies DOWN. "
        "Logistic regression on hand-crafted features achieves AUC 0.528. "
        "FinBERT T+20 reaches AUC 0.523 using only raw text."
    )
    col1, col2 = st.columns(2)
    for fname, title in [
        ("baseline_majority_confusion.png", "Majority Class Baseline"),
        ("baseline_lr_confusion.png",       "Logistic Regression Baseline"),
    ]:
        img = load_image(os.path.join(RESULTS, "baseline", fname))
        if img:
            (col1 if "majority" in fname else col2).image(img, caption=title, use_container_width=True)
    st.markdown("---")
    st.subheader("FinBERT Training Logs")
    for horizon in ["T5","T10","T20"]:
        log = load_csv(os.path.join(RESULTS, "metrics", f"finbert_training_log_{horizon}.csv"))
        if log is not None:
            with st.expander(f"FinBERT {horizon} — Training Log"):
                st.dataframe(log)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("SHAP Feature Importance")
    st.markdown("---")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's contribution "
        "to individual predictions. Analysis run on the XGBoost hybrid ensemble."
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
        "Feature":     ["MACD","RSI","Volume Change","Flesch Reading Ease","Gunning Fog","VADER Sentiment","FinBERT Confidence"],
        "Mean |SHAP|": [0.446, 0.446, 0.319, 0.236, 0.211, 0.173, 0.000],
        "Notes":       [
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
        "FinBERT confidence registers zero SHAP importance because FinBERT inference was run "
        "on test data only. The ablation study (Advanced Analysis) provides independent evidence "
        "that NLP features contribute beyond technical indicators."
    )


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Backtest":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Strategic Backtest")
    st.markdown("---")
    st.markdown(
        "Four strategies simulated on 2023–2025 test set. "
        "20-day hold, £10,000 initial capital, 0.1% transaction cost, 20% max position."
    )
    summary = load_csv(os.path.join(RESULTS, "backtest", "backtest_summary.csv"))
    if summary is not None:
        st.subheader("Strategy Summary")
        st.dataframe(summary, use_container_width=True)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy C — Max Drawdown", "−3.7%",  "vs SPY −18.8%")
    col2.metric("Strategy B — Sortino",      "4.30",   "vs SPY ~1.2")
    col3.metric("Strategy C — Win Rate",     "71.9%")
    col4.metric("Strategy B — Total Return", "+51.9%")
    st.markdown("---")
    st.subheader("Risk-Adjusted Performance")
    comparison_df = pd.DataFrame({
        "Metric":        ["Total return","Max drawdown","Sharpe ratio","Sortino ratio","Win rate","Trades"],
        "Strategy A":    ["+44.2%","−5.2%","1.08","2.03","63.1%","309"],
        "Strategy B":    ["+51.9%","−5.3%","1.68","4.30","67.7%","124"],
        "Strategy C":    ["+43.1%","−3.7%","1.87","4.30","71.9%","89"],
        "Strategy D":    ["+71.5%","−7.0%","1.51","4.51","65.3%","124"],
        "SPY benchmark": ["+82.7%","−18.8%","1.44","~1.2","—","—"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.info(
        "All strategies underperform SPY in absolute return — consistent with semi-strong EMH. "
        "Strategy B Sortino of 4.30 versus SPY ~1.2 indicates substantially more efficient use "
        "of downside risk budget. Strategy C max drawdown −3.7% vs SPY −18.8%."
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


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Custom Backtest":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Custom Backtest Engine")
    st.markdown("---")
    st.markdown(
        "Replay the model's predictions with your own parameters. "
        "Filter by ticker, sector, confidence, and date. Adjust leverage and starting capital."
    )

    trades_path = os.path.join(RESULTS, "backtest", "backtest_trades.csv")
    trades_raw  = load_csv(trades_path)

    if trades_raw is None:
        st.error("backtest_trades.csv not found. Run 12_strategic_backtest.py first.")
        st.stop()

    trades_raw.columns = [c.strip().lower().replace(" ","_") for c in trades_raw.columns]

    col_map = {}
    for needed, candidates in {
        "ticker":     ["ticker","symbol","stock"],
        "entry_date": ["entry_date","date","trade_date","filing_date"],
        "ret":        ["return","ret","trade_return","pnl_pct"],
        "confidence": ["confidence","conf","model_confidence"],
        "strategy":   ["strategy","strat"],
    }.items():
        for c in candidates:
            if c in trades_raw.columns:
                col_map[needed] = c
                break

    if "ticker" in col_map:
        trades_raw["sector"] = trades_raw[col_map["ticker"]].map(
            lambda t: SECTOR_MAP.get(str(t).upper(), "Other")
        )

    if "entry_date" in col_map:
        trades_raw[col_map["entry_date"]] = pd.to_datetime(
            trades_raw[col_map["entry_date"]], errors="coerce"
        )
        date_col = col_map["entry_date"]
        min_date = trades_raw[date_col].min()
        max_date = trades_raw[date_col].max()
    else:
        date_col = min_date = max_date = None

    all_tickers = sorted(trades_raw[col_map["ticker"]].unique()) if "ticker" in col_map else []
    all_sectors = sorted(trades_raw["sector"].unique()) if "sector" in trades_raw.columns else []

    st.subheader("Parameters")
    ctrl1, ctrl2 = st.columns(2)

    with ctrl1:
        st.markdown("**Universe**")
        selection_mode = st.radio(
            "Select tickers by",
            ["All tickers","By sector","Manual selection"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if selection_mode == "All tickers":
            selected_tickers = all_tickers
        elif selection_mode == "By sector":
            selected_sectors = st.multiselect(
                "Sectors", all_sectors,
                default=all_sectors[:2] if len(all_sectors) >= 2 else all_sectors
            )
            selected_tickers = [
                t for t in all_tickers
                if SECTOR_MAP.get(str(t).upper(), "Other") in selected_sectors
            ]
            st.caption(f"{len(selected_tickers)} tickers selected")
        else:
            selected_tickers = st.multiselect(
                "Tickers", all_tickers,
                default=all_tickers[:5] if len(all_tickers) >= 5 else all_tickers
            )

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
        st.markdown("**Capital & Risk**")
        starting_capital = st.number_input(
            "Starting capital (£)", min_value=1000, max_value=1_000_000, value=10_000, step=1000
        )
        leverage = st.slider("Leverage", 1.0, 5.0, 1.0, 0.25,
            help="1x = no leverage. Max drawdown scales with leverage.")
        tx_cost  = st.slider("Transaction cost (%)", 0.0, 1.0, 0.1, 0.05) / 100
        max_pos  = st.slider("Max position size (%)", 5, 50, 20, 5) / 100

        if "confidence" in col_map:
            conf_min = float(trades_raw[col_map["confidence"]].min())
            conf_max = float(trades_raw[col_map["confidence"]].max())
            conf_threshold = st.slider(
                "Min confidence threshold",
                min_value=round(conf_min,2), max_value=round(conf_max,2),
                value=round(conf_min,2), step=0.01
            )
        else:
            conf_threshold = 0.0

    # ── filter ────────────────────────────────────────────────────────────────
    df = trades_raw.copy()
    if "ticker" in col_map and selected_tickers:
        df = df[df[col_map["ticker"]].isin(selected_tickers)]
    if date_col and date_range and len(date_range) == 2:
        s, e = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df[date_col] >= s) & (df[date_col] <= e)]
    if "confidence" in col_map:
        df = df[df[col_map["confidence"]] >= conf_threshold]

    if len(df) == 0:
        st.warning("No trades match the selected filters. Try widening your selection.")
        st.stop()

    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)

    ret_col = col_map.get("ret")
    if ret_col and ret_col in df.columns:
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce").fillna(0)

        capital      = starting_capital
        equity_curve = [capital]
        dates_curve  = [df[date_col].iloc[0]] if date_col else []

        for _, row in df.iterrows():
            pos   = min(capital * max_pos, capital) * leverage
            tr    = float(row[ret_col]) * leverage - tx_cost
            capital = max(capital + pos * tr, 0)
            equity_curve.append(capital)
            if date_col:
                dates_curve.append(row[date_col])

        total_return = (equity_curve[-1] - starting_capital) / starting_capital * 100
        n_trades     = len(df)
        win_rate     = (df[ret_col] > 0).mean() * 100

        peak = starting_capital
        max_dd = 0.0
        for v in equity_curve:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak * 100 if peak > 0 else 0)

        rets = pd.Series([
            (equity_curve[i+1] - equity_curve[i]) / equity_curve[i]
            for i in range(len(equity_curve)-1) if equity_curve[i] > 0
        ])
        down = rets[rets < 0]
        sortino = (rets.mean() / down.std() * np.sqrt(252)
                   if len(down) > 1 and down.std() > 0 else 0.0)

        st.markdown("---")
        st.subheader("Results")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Total Return",  f"{total_return:+.1f}%")
        m2.metric("Final Capital", f"£{equity_curve[-1]:,.0f}")
        m3.metric("Trades",        str(n_trades))
        m4.metric("Win Rate",      f"{win_rate:.1f}%")
        m5.metric("Max Drawdown",  f"−{max_dd:.1f}%")
        m6.metric("Sortino",       f"{sortino:.2f}")

        st.markdown("---")
        st.subheader("Equity Curve")
        if dates_curve and len(dates_curve) == len(equity_curve):
            eq_df = pd.DataFrame({"Capital": equity_curve}, index=dates_curve)
        else:
            eq_df = pd.DataFrame({"Capital": equity_curve})
        st.line_chart(eq_df, use_container_width=True)

        st.subheader("Drawdown")
        eq_s   = pd.Series(equity_curve)
        dd_s   = (eq_s - eq_s.cummax()) / eq_s.cummax() * 100
        dd_df  = pd.DataFrame({"Drawdown (%)": dd_s})
        if dates_curve and len(dates_curve) == len(dd_s):
            dd_df.index = dates_curve
        st.area_chart(dd_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Trade Log")
        show_cols = [c for c in [
            col_map.get("ticker"), date_col, ret_col,
            col_map.get("confidence"), col_map.get("strategy"), "sector"
        ] if c and c in df.columns]
        disp = df[show_cols].copy()
        if ret_col in disp.columns:
            disp[ret_col] = disp[ret_col].map(lambda x: f"{x:+.2%}")
        if col_map.get("confidence") in disp.columns:
            disp[col_map["confidence"]] = disp[col_map["confidence"]].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )
        st.dataframe(disp, use_container_width=True, hide_index=True)

        if "ticker" in col_map:
            st.markdown("---")
            st.subheader("Per-Ticker Summary")
            tg = df.groupby(col_map["ticker"])[ret_col].agg(
                Trades="count",
                Win_Rate=lambda x: f"{(x>0).mean()*100:.0f}%",
                Avg_Return=lambda x: f"{x.mean()*100:+.2f}%",
                Total_Return=lambda x: f"{x.sum()*100:+.2f}%",
            ).reset_index()
            tg.columns = ["Ticker","Trades","Win Rate","Avg Return","Total Return"]
            st.dataframe(tg, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not identify a return column in backtest_trades.csv.")
        st.write("Columns found:", list(trades_raw.columns))


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Error Analysis":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Error Analysis")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy",    "56.0%")
    col2.metric("False Positive Rate", "27.7%", "Predicted UP — stock fell")
    col3.metric("False Negative Rate", "16.3%", "Predicted DOWN — stock rose")
    st.markdown("---")
    for fname, title, caption in [
        ("error_breakdown.png",        "Error Breakdown & Sector Accuracy",
         "Accuracy, false positive, and false negative rates by sector."),
        ("error_by_quarter.png",       "Accuracy by Quarter",
         "Temporal breakdown of prediction accuracy across 2023–2025."),
        ("fp_return_distribution.png", "False Positive Return Distribution",
         "Actual returns on filings where the model predicted UP but the stock fell."),
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
        df_e = load_csv(os.path.join(RESULTS, "error_analysis", fname))
        if df_e is not None:
            col.subheader(title)
            col.dataframe(df_e, use_container_width=True)
    st.markdown("---")
    st.subheader("Failure Pattern Summary")
    patterns_df = pd.DataFrame({
        "Pattern": [
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


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Advanced Analysis":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Advanced Analysis")
    st.markdown("---")
    adv_path = os.path.join(RESULTS, "advanced")
    adv_sections = [
        ("A_calibration_plot.png",      "A. Calibration (Reliability Diagram)",
         "ECE = 0.049 — well calibrated. Predicted probability tracks actual accuracy."),
        ("B_caar_event_study.png",       "B. CAAR Event Study",
         "UP-labelled filings outperform DOWN-labelled by +11.81pp by day +20, independently validating label quality."),
        ("C_sector_breakdown.png",       "C. Per-Sector Performance",
         "Accuracy across all 8 sectors. Financials 73.6%; Utilities 40.0%."),
        ("D_confidence_sensitivity.png", "D. Confidence Threshold Sensitivity",
         "Accuracy vs. trade count at each threshold. Optimal 0.72 used in Strategy B."),
        ("E_roc_curves.png",             "E. ROC Curves — T+5, T+10, T+20",
         "AUC: 0.498 → 0.515 → 0.523. Monotonic improvement supports gradual diffusion hypothesis."),
        ("F_length_vs_accuracy.png",     "F. Filing Length vs Prediction Accuracy",
         "Whether chunk count correlates with accuracy."),
        ("G_significance_tests.png",     "G. Statistical Significance Tests",
         "Binomial p=0.000032. Accuracy p=0.0069. McNemar p=0.10 (not significant — acknowledged)."),
        ("H_ablation_study.png",         "H. Ablation Study",
         "NLP only 54.9% vs Technical only 53.0% vs All features 57.7%."),
        ("I_finbert_confusion_matrices.png","I. FinBERT Confusion Matrices — All Horizons",
         "Per-class breakdown across T+5, T+10, T+20 models."),
    ]
    for fname, title, caption in adv_sections:
        img = load_image(os.path.join(adv_path, fname))
        if img:
            st.subheader(title)
            st.caption(caption)
            st.image(img, use_container_width=True)
            st.markdown("---")
        else:
            st.caption(f"{title} — run 14_advanced_analysis.py to generate.")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Prediction":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Live Prediction")
    st.markdown("---")
    st.markdown(
        "Paste text from a real SEC 10-K or 10-Q filing below. "
        "The fine-tuned **FinBERT T+20 model** will predict whether the stock is likely "
        "to go UP or DOWN in the 20 trading days (~1 month) following the filing date."
    )
    st.info(
        "Model achieves 60.9% filing-level accuracy and 63.1% UP win rate on the 2023–2025 test set. "
        "Predictions are based on filing language only — macroeconomic conditions are not considered."
    )

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
                ids      = [tokenizer.cls_token_id] + tc + [tokenizer.sep_token_id]
                ids      = ids[:512]
                pad      = [tokenizer.pad_token_id] * (512 - len(ids))
                iids     = torch.tensor([ids + pad])
                mask     = torch.tensor([[1]*len(ids) + [0]*len(pad)])
                out      = model(input_ids=iids, attention_mask=mask)
                probs    = torch.softmax(out.logits, dim=1)[0]
                prob_up  = float(probs[1])
                results.append({"n_tokens": len(tc), "prob_up": round(prob_up,4), "pred": int(prob_up >= 0.5)})
        n_up  = sum(r["pred"] for r in results)
        n_down = len(results) - n_up
        conf  = sum(r["prob_up"] for r in results) / len(results)
        return {"prediction": int(n_up >= n_down), "confidence": conf,
                "n_chunks": len(results), "n_up_votes": n_up,
                "n_down_votes": n_down, "chunk_details": results}

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

    text_input = st.text_area(
        "Paste SEC filing text (MD&A, Risk Factors, etc.):",
        value=st.session_state.get("live_text",""),
        height=250,
        key="live_text_area",
        placeholder="e.g. paste a paragraph from Item 7 (Management Discussion and Analysis)...",
    )

    if st.button("Run T+20 Prediction", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please paste some filing text first.")
        else:
            with st.spinner("Running FinBERT T+20 inference..."):
                tok, mdl = load_finbert_model()
            if tok is None:
                st.error("FinBERT T+20 model not found at 03_Models/finbert_champion_T20/. Run 08_train_finbert.py --horizon T20 first.")
            else:
                with st.spinner("Analysing chunks..."):
                    result = predict_text(text_input, tok, mdl)
                st.markdown("---")
                direction = "UP" if result["prediction"] == 1 else "DOWN"
                if result["prediction"] == 1:
                    st.success(f"Predicted direction: **{direction}** — positive price movement expected over T+20")
                else:
                    st.error(f"Predicted direction: **{direction}** — negative price movement expected over T+20")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Prediction",     direction)
                c2.metric("Avg Confidence", f"{result['confidence']:.1%}")
                c3.metric("Text Chunks",    result["n_chunks"])
                c4.metric("Vote Split",     f"{result['n_up_votes']} UP / {result['n_down_votes']} DOWN")
                st.markdown("**Chunk-level detail:**")
                cdf = pd.DataFrame(result["chunk_details"])
                cdf.index    = [f"Chunk {i+1}" for i in range(len(cdf))]
                cdf["vote"]  = cdf["pred"].map({1:"UP",0:"DOWN"})
                cdf["prob_up"] = cdf["prob_up"].map(lambda x: f"{x:.1%}")
                cdf = cdf.rename(columns={"n_tokens":"Tokens","prob_up":"P(UP)","vote":"Vote"})
                st.dataframe(cdf[["Tokens","P(UP)","Vote"]], use_container_width=True)
                conf = result["confidence"]
                strength = "high" if abs(conf-0.5)>0.15 else "moderate" if abs(conf-0.5)>0.07 else "low"
                st.caption(f"Average chunk confidence: {conf:.1%} ({strength} conviction). Prediction based on filing language only.")

    st.markdown("---")
    st.caption("Model: ProsusAI/finbert fine-tuned on SEC 10-K/10-Q filings, T+20 label. Trained pre-2021, validated 2021–2022, tested 2023+.")