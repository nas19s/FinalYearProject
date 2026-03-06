import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, roc_auc_score, accuracy_score)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, '01_Data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, '04_Results', 'metrics')
MODELS_DIR = os.path.join(PROJECT_ROOT, '03_Models', 'hybrid_ensemble')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_COL = 'Label_T20'
HORIZON = 'T20'

# Section weighting mapping
SECTION_WEIGHTS = {
    "item_1a_risk_factors": 1.0,
    "item_7_mda": 1.0,
    "item_7a_market_risk": 0.6,
    "item_9a_controls": 0.8,
}

FEATURE_COLS = [
    "RSI", "MACD", "Volume_Change",
    "Gunning_Fog", "Flesch_Ease", "Sentiment",
    "Diff_Word_Ratio", "Word_Count",
]

def weighted_vote(group):
    """Calculates weighted prediction for a filing based on section importance."""
    total_weight_up = 0.0
    total_weight_down = 0.0

    for _, row in group.iterrows():
        w = SECTION_WEIGHTS.get(row["section"], 0.8)
        if row["pred_label"] == 1:
            total_weight_up += w * row["prob_up"]
        else:
            total_weight_down += w * (1 - row["prob_up"])

    total = total_weight_up + total_weight_down
    confidence = total_weight_up / total if total > 0 else 0.5
    prediction = 1 if confidence >= 0.5 else 0

    return {
        "pred_label": prediction,
        "confidence": confidence,
        "true_label": group["true_label"].iloc[0],
        "weight_up": total_weight_up,
        "weight_down": total_weight_down,
    }

def main():
    print('Starting section-weighted voting ensemble...')

    preds_path = os.path.join(RESULTS_DIR, f"finbert_full_test_predictions_{HORIZON}.csv")
    preds_df = pd.read_csv(preds_path)
    
    # Filing-level aggregation
    results = []
    for (ticker, filing_date), group in preds_df.groupby(["ticker", "filing_date"]):
        vote = weighted_vote(group)
        vote["ticker"] = ticker
        vote["filing_date"] = filing_date
        vote["n_sections"] = group["section"].nunique()
        vote["n_chunks"] = len(group)
        results.append(vote)

    voting_df = pd.DataFrame(results)
    
    # Metrics
    y_true = voting_df["true_label"].astype(int)
    y_pred = voting_df["pred_label"].astype(int)
    y_prob = voting_df["confidence"].astype(float)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_down = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    f1_up = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")

    print(f'Ensemble Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')

    # Breakdown by section
    for section, weight in SECTION_WEIGHTS.items():
        sec_chunks = preds_df[preds_df["section"] == section]
        if len(sec_chunks) == 0: continue
        s_acc = accuracy_score(sec_chunks["true_label"], sec_chunks["pred_label"])
        s_f1 = f1_score(sec_chunks["true_label"], sec_chunks["pred_label"], average="macro", zero_division=0)
        print(f"Section: {section:<25} Acc: {s_acc:.3f} F1: {s_f1:.3f}")

    # XGBoost Hybrid Model
    xgb_results = None
    if XGB_AVAILABLE:
        print("\nTraining XGBoost on numerical features + FinBERT confidence...")
        
        feat_path = os.path.join(DATA_DIR, "final_feature_dataset.parquet")
        df = pd.read_parquet(feat_path)
        df = df[df[TARGET_COL] != 0].copy()
        df["binary_label"] = df[TARGET_COL].map({-1: 0, 1: 1})
        df = df.dropna(subset=["binary_label"] + FEATURE_COLS)
        df["filing_date"] = pd.to_datetime(df["filing_date"])

        train_df = df[df["filing_date"] < "2021-01-01"]
        test_df = df[df["filing_date"] >= "2023-01-01"]

        agg_dict = {col: "mean" for col in FEATURE_COLS}
        agg_dict["binary_label"] = "first"
        
        train_agg = train_df.groupby(["ticker", "filing_date"]).agg(agg_dict).reset_index()
        test_agg = test_df.groupby(["ticker", "filing_date"]).agg(agg_dict).reset_index()

        voting_df["filing_date"] = pd.to_datetime(voting_df["filing_date"])
        test_agg = test_agg.merge(voting_df[["ticker", "filing_date", "confidence"]], on=["ticker", "filing_date"], how="left")
        test_agg["confidence"] = test_agg["confidence"].fillna(0.5)
        train_agg["confidence"] = 0.5

        XGB_COLS = FEATURE_COLS + ["confidence"]
        X_train, y_train = train_agg[XGB_COLS].fillna(0).values, train_agg["binary_label"].astype(int).values
        X_test, y_test = test_agg[XGB_COLS].fillna(0).values, test_agg["binary_label"].astype(int).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_down, n_up = (y_train == 0).sum(), (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_up/n_down,
            random_state=42, eval_metric="logloss"
        )
        xgb.fit(X_train_s, y_train)

        yp, yprob = xgb.predict(X_test_s), xgb.predict_proba(X_test_s)[:, 1]
        x_acc = accuracy_score(y_test, yp)
        x_f1 = f1_score(y_test, yp, average="macro", zero_division=0)
        
        try:
            x_auc = roc_auc_score(y_test, yprob)
        except:
            x_auc = float("nan")

        print(f"XGBoost Hybrid -> Accuracy: {x_acc:.4f} F1: {x_f1:.4f}")

        xgb_results = {
            "Model": "XGBoost + FinBERT Voting Confidence",
            "Accuracy": round(x_acc, 4), "F1_Macro": round(x_f1, 4), "AUC": round(x_auc, 4),
            "F1_DOWN": round(f1_score(y_test, yp, pos_label=0, average="binary"), 4),
            "F1_UP": round(f1_score(y_test, yp, pos_label=1, average="binary"), 4),
            "N_test": len(y_test)
        }

        # Save artifacts
        pickle.dump(xgb, open(os.path.join(MODELS_DIR, "hybrid_rf_model.pkl"), "wb"))
        pickle.dump(scaler, open(os.path.join(MODELS_DIR, "hybrid_scaler.pkl"), "wb"))
        pickle.dump(XGB_COLS, open(os.path.join(MODELS_DIR, "hybrid_feature_cols.pkl"), "wb"))

    # Exports
    voting_df.to_csv(os.path.join(RESULTS_DIR, "voting_ensemble_predictions.csv"), index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    plt.title("Ensemble Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "hybrid_confusion_matrix.png"), dpi=150)
    plt.close()

    # Update results registry
    master_path = os.path.join(RESULTS_DIR, "master_results_table.csv")
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        master = master[~master["Model"].str.contains("Hybrid|Stacking|Voting|XGBoost", na=False)]
        
        voting_entry = pd.DataFrame([{
            "Model": "FinBERT Section-Weighted Voting",
            "Accuracy": round(acc, 4), "F1_Macro": round(f1, 4), "AUC": round(auc, 4),
            "F1_DOWN": round(f1_down, 4), "F1_UP": round(f1_up, 4), "N_test": len(voting_df)
        }])
        
        entries = [master, voting_entry]
        if xgb_results: entries.append(pd.DataFrame([xgb_results]))
        
        final_table = pd.concat(entries, ignore_index=True).sort_values("F1_Macro", ascending=False)
        final_table.to_csv(os.path.join(RESULTS_DIR, "master_results_table_final.csv"), index=False)
        print("\nFinal Performance Comparison:")
        print(final_table.to_string(index=False))

    print('Processing complete.')

if __name__ == '__main__':
    main()