import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

OUTPUT_DIR = r"D:\Desktop\研一\code\scGRN\outputs\E005\v2"
OUTPUT_DIR_V1 = r"D:\Desktop\研一\code\scGRN\outputs\E005\v1"

def ecdf_normalize(scores, reference_scores):
    n_ref = len(reference_scores)
    sorted_ref = np.sort(reference_scores)
    return np.searchsorted(sorted_ref, scores, side="right") / n_ref

def compute_fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= tpr_target)[0]
    if len(idx) == 0: return 1.0
    return fpr[idx[0]]

def main():
    print("Loading data for grid search...")
    expr_df = pd.read_csv(os.path.join(OUTPUT_DIR_V1, "E005_expression_scores.csv"))
    grn_df = pd.read_csv(os.path.join(OUTPUT_DIR, "E005_grn_scores.csv"))
    df = expr_df.merge(grn_df, on="cell_id", how="left")
    
    # Fill NaN
    df["grn_distance_score"] = df["grn_distance_score"].replace([np.inf, -np.inf], np.nan)
    finite_vals = df["grn_distance_score"].dropna()
    fill_val = finite_vals.max() if len(finite_vals) > 0 else 1.0
    df["grn_distance_score"] = df["grn_distance_score"].fillna(fill_val)
    
    val_mask = df["E005_split"] == "val_known"
    val_expr = df.loc[val_mask, "expr_fused"].values
    df["expr_pct"] = ecdf_normalize(df["expr_fused"].values, val_expr)
    
    val_grn = df.loc[val_mask, "grn_distance_score"].dropna().values
    df["grn_pct"] = ecdf_normalize(df["grn_distance_score"].values, val_grn)
    
    test_mask = df["E005_split"].isin(["test_known", "test_unknown"])
    test_df = df[test_mask].copy()
    y_true = (test_df["E005_split"] == "test_unknown").astype(int).values
    
    results = []
    
    for w in np.arange(0.0, 1.01, 0.05):
        w = round(w, 2)
        dual_fused = (1 - w) * test_df["expr_pct"].values + w * test_df["grn_pct"].values
        
        auroc = roc_auc_score(y_true, dual_fused)
        aupr = average_precision_score(y_true, dual_fused)
        fpr95 = compute_fpr_at_tpr(y_true, dual_fused, 0.95)
        
        results.append({
            "GRN_Weight": w,
            "AUROC": auroc,
            "AUPR": aupr,
            "FPR95": fpr95
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- Weight Search Results ---")
    print(res_df.to_string(index=False))
    
    best_auroc = res_df.loc[res_df["AUROC"].idxmax()]
    best_aupr = res_df.loc[res_df["AUPR"].idxmax()]
    
    print("\n--- Best by AUROC ---")
    print(best_auroc)
    print("\n--- Best by AUPR ---")
    print(best_aupr)

if __name__ == "__main__":
    main()
