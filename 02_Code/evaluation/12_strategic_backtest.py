import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "backtest")
METRICS_DIR = os.path.join(PROJECT_ROOT, "04_Results", "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Simulation Parameters
HOLD_DAYS = 20
INITIAL_CAP = 10000.0
MAX_POSITION = 0.20
TRANSACTION_COST = 0.001
CONF_THRESHOLD = 0.72
MARGIN_RATE = 0.02 / 252
STOP_LOSS_D = -0.08

SECTOR_MAP = {
    "AAPL": "Technology",   "MSFT": "Technology",  "NVDA": "Technology",
    "GOOGL": "Technology",  "META": "Technology",   "AVGO": "Technology",
    "ORCL": "Technology",   "CRM": "Technology",    "AMD": "Technology",
    "INTC": "Technology",   "QCOM": "Technology",   "TXN": "Technology",
    "AMZN": "Consumer",     "TSLA": "Consumer",     "HD": "Consumer",
    "MCD": "Consumer",      "NKE": "Consumer",      "SBUX": "Consumer",
    "WMT": "Consumer",      "COST": "Consumer",      "TGT": "Consumer",
    "JPM": "Financials",    "BAC": "Financials",    "WFC": "Financials",
    "GS": "Financials",     "MS": "Financials",     "BRK-B": "Financials",
    "V": "Financials",      "MA": "Financials",     "AXP": "Financials",
    "JNJ": "Healthcare",    "UNH": "Healthcare",    "PFE": "Healthcare",
    "ABBV": "Healthcare",   "MRK": "Healthcare",    "ABT": "Healthcare",
    "LLY": "Healthcare",    "TMO": "Healthcare",
    "XOM": "Energy",        "CVX": "Energy",        "COP": "Energy",
    "NEE": "Utilities",     "DUK": "Utilities",
    "CAT": "Industrials",   "HON": "Industrials",   "UPS": "Industrials",
    "BA": "Industrials",    "RTX": "Industrials",
    "LIN": "Materials",     "APD": "Materials",
}

GOOD_SECTORS = {"Financials", "Technology"}

def load_prices(ticker):
    for suffix in ("_prices.csv", ".csv"):
        path = os.path.join(PRICES_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                df = df.sort_values("Date").set_index("Date")
                df.index = pd.to_datetime(df.index)
                return df
            except Exception:
                return None
    return None

def load_spy():
    for name in ("SPY_prices.csv", "SPY.csv", "spy_prices.csv", "spy.csv"):
        path = os.path.join(PRICES_DIR, name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["Date"])
                df = df.sort_values("Date").set_index("Date")
                df.index = pd.to_datetime(df.index)
                return df
            except Exception:
                continue
    return None

def get_trade_return(prices, entry_date, stop_loss=None):
    future = prices[prices.index > entry_date].head(HOLD_DAYS + 10)
    if len(future) < 2:
        return 0.0, 0

    entry_price = future.iloc[0]["Open"]
    if pd.isna(entry_price) or entry_price <= 0:
        return 0.0, 0

    for i in range(1, min(HOLD_DAYS + 1, len(future))):
        row = future.iloc[i]
        raw_ret = (row["Open"] - entry_price) / entry_price

        if stop_loss is not None and raw_ret <= stop_loss:
            return stop_loss - TRANSACTION_COST, i

        if i == HOLD_DAYS or i == len(future) - 1:
            return raw_ret - TRANSACTION_COST, i
    return 0.0, 0

def run_backtest(signals, name, leveraged=False):
    signals = signals.sort_values("filing_date").reset_index(drop=True)
    trades = []
    capital = INITIAL_CAP
    open_trades = []

    for _, row in signals.iterrows():
        ticker = row["ticker"]
        entry_date = pd.to_datetime(row["filing_date"])
        prices = load_prices(ticker)
        if prices is None:
            continue

        still_open = []
        for t in open_trades:
            if t["exit_date"] <= entry_date:
                capital += t["allocated"] * t["return"]
            else:
                still_open.append(t)
        open_trades = still_open

        if leveraged:
            position_size = min(capital * MAX_POSITION * 2, capital * 0.40)
        else:
            n_open = len(open_trades) + 1
            position_size = min(capital * MAX_POSITION, capital / n_open)

        position_size = max(position_size, 0)
        if position_size < 10:
            continue

        stop = STOP_LOSS_D if leveraged else None
        ret, days = get_trade_return(prices, entry_date, stop_loss=stop)

        if leveraged:
            ret -= MARGIN_RATE * days

        future_dates = prices[prices.index > entry_date].head(days + 2)
        if len(future_dates) > days:
            exit_date = future_dates.index[days]
        else:
            exit_date = entry_date + pd.Timedelta(days=int(days * 1.4))

        capital -= position_size
        open_trades.append({
            "exit_date": exit_date,
            "return": 1 + ret,
            "allocated": position_size,
        })

        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "return": ret,
            "days_held": days,
            "allocated": position_size,
            "strategy": name,
        })

    for t in open_trades:
        capital += t["allocated"] * t["return"]

    return pd.DataFrame(trades), capital

def compute_equity_curve(trades_df, initial_cap=10000.0):
    if trades_df.empty:
        return pd.Series([initial_cap])
    t = trades_df.sort_values("entry_date").reset_index(drop=True)
    capital = initial_cap
    curve = [initial_cap]
    for _, row in t.iterrows():
        capital += row["allocated"] * row["return"]
        curve.append(capital)
    return pd.Series(curve)

def compute_metrics(trades_df, final_capital, name):
    r = trades_df["return"]
    eq = compute_equity_curve(trades_df)
    win_rate = (r > 0).mean()
    avg_ret = r.mean()
    total_r = (final_capital - INITIAL_CAP) / INITIAL_CAP
    
    sharpe = (r.mean() / r.std() * np.sqrt(252 / HOLD_DAYS) if r.std() > 0 else 0.0)
    
    downside = r[r < 0]
    down_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = (r.mean() / down_std * np.sqrt(252 / HOLD_DAYS) if down_std > 0 else 0.0)
    
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min()

    print(f"\nResults for {name}:")
    print(f"  Total return: {total_r:.2%}")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Max Drawdown: {mdd:.2%}")

    return {
        "strategy": name,
        "trades": len(r),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_ret, 4),
        "total_return": round(total_r, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 4),
        "final_equity": round(final_capital, 2),
    }

def main():
    print("Starting backtest execution...")
    preds = pd.read_csv(os.path.join(METRICS_DIR, "voting_ensemble_predictions.csv"))
    preds["filing_date"] = pd.to_datetime(preds["filing_date"]).dt.tz_localize(None)
    preds["sector"] = preds["ticker"].map(SECTOR_MAP).fillna("Other")
    preds = preds.sort_values("filing_date").reset_index(drop=True)

    # Strategy A
    signals_a = preds[preds["pred_label"] == 1].copy()
    trades_a, final_a = run_backtest(signals_a, "Strategy A (all UP)")

    # Strategy B
    signals_b = preds[(preds["pred_label"] == 1) & (preds["confidence"] >= CONF_THRESHOLD)].copy()
    trades_b, final_b = run_backtest(signals_b, f"Strategy B (conf>={CONF_THRESHOLD})")

    # Strategy C
    signals_c = preds[(preds["pred_label"] == 1) & (preds["sector"].isin(GOOD_SECTORS))].copy()
    trades_c, final_c = run_backtest(signals_c, "Strategy C (Fin+Tech only)")

    # Strategy D
    trades_d, final_d = run_backtest(signals_b.copy(), "Strategy D (2x leveraged)", leveraged=True)

    results = []
    eq_curves = {}
    
    backtest_data = [
        (trades_a, final_a, "Strategy A: All UP signals"),
        (trades_b, final_b, f"Strategy B: High Confidence (>={CONF_THRESHOLD})"),
        (trades_c, final_c, "Strategy C: Sector-Filtered (Fin+Tech)"),
        (trades_d, final_d, "Strategy D: 2x Leveraged"),
    ]

    for trades, final, name in backtest_data:
        if not trades.empty:
            results.append(compute_metrics(trades, final, name))
            eq_curves[name] = compute_equity_curve(trades)

    # Benchmark Comparison
    spy = load_spy()
    if spy is not None:
        start, end = preds["filing_date"].min(), preds["filing_date"].max()
        spy_p = spy[(spy.index >= start) & (spy.index <= end)]["Close"]
        if len(spy_p) > 1:
            spy_ret = (spy_p.iloc[-1] - spy_p.iloc[0]) / spy_p.iloc[0]
            spy_daily = spy_p.pct_change().dropna()
            spy_eq_raw = (1 + spy_daily).cumprod() * INITIAL_CAP
            spy_eq = pd.concat([pd.Series([INITIAL_CAP]), spy_eq_raw], ignore_index=True)
            results.append({
                "strategy": "SPY Benchmark",
                "total_return": round(spy_ret, 4),
                "final_equity": round(INITIAL_CAP * (1 + spy_ret), 2)
            })

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    for name, eq in eq_curves.items():
        ax.plot(eq.values, label=name, linewidth=2)
    ax.set_title("Portfolio Equity Curves")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    for label, trades in [("A", trades_a), ("B", trades_b), ("C", trades_c), ("D", trades_d)]:
        if not trades.empty:
            ax2.hist(trades["return"], bins=25, alpha=0.5, label=label)
    ax2.set_title("Return Distribution")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "backtest_equity_curves.png"))

    pd.concat([trades_a, trades_b, trades_c, trades_d]).to_csv(os.path.join(RESULTS_DIR, "backtest_trades.csv"), index=False)
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "backtest_summary.csv"), index=False)
    print("Backtest complete. Files saved.")

if __name__ == "__main__":
    main()