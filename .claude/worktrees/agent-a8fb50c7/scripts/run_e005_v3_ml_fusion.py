"""
E005 Step 5 (V3): ML-based Anomaly Detection Fusion
=====================================================================
1. Load expression features (entropy_norm, distance_norm) from v1
2. Load full AUCell matrix from v2
3. Select Top N most variable regulons (based on known centroids)
4. Train Isolation Forest on train_known to establish "known cell profile"
5. Score all cells (Anomaly Score) -> invert so higher = unknown
6. Evaluate and output to v3/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = r"D:\Desktop\研一\code\scGRN\outputs\E005\v3"
OUTPUT_DIR_V1 = r"D:\Desktop\研一\code\scGRN\outputs\E005\v1"
OUTPUT_DIR_V2 = r"D:\Desktop\研一\code\scGRN\outputs\E005\v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_CLASSES = ["cDC2", "pDC", "cDC1"]
UNKNOWN_CLASS = "ASDC"

N_TOP_REGULONS = 50

def log(msg):
    print(f"[E005-V3] {msg}", flush=True)

def compute_fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= tpr_target)[0]
    if len(idx) == 0: return 1.0
    return fpr[idx[0]]

def main():
    # 1. Load Data
    log("Loading data...")
    expr_df = pd.read_csv(os.path.join(OUTPUT_DIR_V1, "E005_expression_scores.csv"))
    auc_df = pd.read_csv(os.path.join(OUTPUT_DIR_V2, "E005_aucell_all_cells.csv"), index_col=0)
    
    # 2. Select Top N regulons
    log(f"Selecting Top {N_TOP_REGULONS} variable regulons...")
    train_labels = expr_df[expr_df["E005_split"] == "train_known"].set_index("cell_id")["true_label"]
    centroids = {}
    for cls in KNOWN_CLASSES:
        cls_cells = train_labels[train_labels == cls].index
        avail = [c for c in cls_cells if c in auc_df.index]
        centroids[cls] = auc_df.loc[avail].mean(axis=0)
    
    centroid_df = pd.DataFrame(centroids).T
    regulon_var = centroid_df.var(axis=0)
    top_regulons = regulon_var.nlargest(N_TOP_REGULONS).index.tolist()
    log(f"  Top 5 regulons: {top_regulons[:5]}")
    
    # 3. Build Feature Matrix
    log("Building hybrid feature matrix...")
    df = expr_df[["cell_id", "E005_split", "true_label", "nearest_known_class", "entropy_norm", "distance_norm", "expr_fused"]].copy()
    
    auc_top = auc_df[top_regulons].copy()
    auc_top["cell_id"] = auc_top.index
    df = df.merge(auc_top, on="cell_id", how="left")
    
    feature_cols = ["entropy_norm", "distance_norm"] + top_regulons
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    # 4. Train Isolation Forest
    log("Training Isolation Forest on train_known features...")
    train_mask = df["E005_split"] == "train_known"
    X_train = df.loc[train_mask, feature_cols].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = IsolationForest(
        n_estimators=200, 
        max_samples="auto", 
        contamination="auto", 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train_scaled)
    
    # 5. Score all cells
    log("Scoring all cells via Isolation Forest...")
    X_all = df[feature_cols].values
    X_all_scaled = scaler.transform(X_all)
    
    # Invert decision function: lower means anomaly -> -clf.decision_function makes larger = anomaly
    df["if_anomaly_score"] = -clf.decision_function(X_all_scaled)
    
    # 6. Evaluation
    log("Evaluating on Test Set...")
    test_mask = df["E005_split"].isin(["test_known", "test_unknown"])
    test_df = df[test_mask]
    
    y_true = (test_df["E005_split"] == "test_unknown").astype(int).values
    
    metrics = {}
    score_kinds = {
        "Baseline (expr_fused)": "expr_fused",
        "V3_ML_Anomaly": "if_anomaly_score"
    }
    
    for name, col in score_kinds.items():
        scores = test_df[col].values
        auroc = roc_auc_score(y_true, scores)
        aupr = average_precision_score(y_true, scores)
        fpr95 = compute_fpr_at_tpr(y_true, scores, 0.95)
        metrics[name] = {
            "AUROC": round(auroc, 4),
            "AUPR": round(aupr, 4),
            "FPR95": round(fpr95, 4)
        }
        log(f"  {name}: AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR95={fpr95:.4f}")
        
    df.to_csv(os.path.join(OUTPUT_DIR, "E005_v3_fusion_scores.csv"), index=False)
    
    # Plot ROC
    fig, ax = plt.subplots(figsize=(8,6))
    for name, col in score_kinds.items():
        scores = test_df[col].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['AUROC']:.4f})", linewidth=2)
    ax.plot([0,1], [0,1], "k--", alpha=0.5)
    ax.set_title("E005 V3: Isolation Forest Anomaly Detection ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "E005_v3_roc.png"), dpi=150)
    plt.close()
    
    # Save Summary
    with open(os.path.join(OUTPUT_DIR, "E005_v3_summary.md"), "w", encoding="utf-8") as f:
        f.write("# E005 V3 Isolation Forest Results\n\n")
        f.write(f"- Selected Top {N_TOP_REGULONS} variable Regulons\n")
        f.write(f"- Features used: `entropy_norm`, `distance_norm` + `{N_TOP_REGULONS} Regulons`\n\n")
        f.write("## Unknown Detection Metrics\n\n")
        f.write("| Method | AUROC | AUPR | FPR95 |\n")
        f.write("|---|---|---|---|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v['AUROC']} | {v['AUPR']} | {v['FPR95']} |\n")
            
    with open(os.path.join(OUTPUT_DIR, "E005_v3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log("=" * 60)
    log("V3 COMPLETE!")
    log("=" * 60)

if __name__ == "__main__":
    main()
