# E005 Step 3: pySCENIC CLI pipeline
# ====================================
# Runs the 3 pySCENIC steps: grn -> ctx -> aucell
# 
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_e005_pyscenic_cli.ps1
#
# Prerequisites:
#   - pyscenic installed (pip install pyscenic)
#   - Step 2 completed (loom file exists)

$ErrorActionPreference = "Stop"

# --- Paths ---
$PROJECT_DIR = "D:\Desktop\研一\code\scGRN"
$OUTPUT_DIR  = "$PROJECT_DIR\outputs\E005"
$DATA_DIR    = "$PROJECT_DIR\data\pyscenic_dbs"

$EXPR_LOOM   = "$OUTPUT_DIR\E005_train_known_expr.loom"
$TF_LIST     = "$DATA_DIR\allTFs_hg38.txt"
$RANKING_DB  = "$DATA_DIR\hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
$MOTIF_TBL   = "$DATA_DIR\motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"

# --- Output files ---
$GRN_ADJ     = "$OUTPUT_DIR\E005_grn_adjacencies.tsv"
$CTX_REG     = "$OUTPUT_DIR\E005_regulons.csv"
$AUCELL_LOOM = "$OUTPUT_DIR\E005_aucell.loom"

Write-Host "============================================================"
Write-Host "[E005-pyscenic] Starting pySCENIC pipeline"
Write-Host "============================================================"

# Check inputs exist
if (-not (Test-Path $EXPR_LOOM)) {
    Write-Host "[ERROR] Expression loom not found: $EXPR_LOOM"
    Write-Host "        Run run_e005_export_pyscenic_input.py first."
    exit 1
}

# --- Step 1: GRN (co-expression modules) ---
Write-Host ""
Write-Host "[Step 1/3] pyscenic grn ..."
$grn_start = Get-Date

pyscenic grn `
    $EXPR_LOOM `
    $TF_LIST `
    -o $GRN_ADJ `
    --num_workers 8 `
    --method grnboost2

$grn_end = Get-Date
Write-Host "[Step 1/3] GRN done. Time: $(($grn_end - $grn_start).TotalMinutes.ToString('F1')) min"
Write-Host "  Output: $GRN_ADJ"

# --- Step 2: CTX (motif enrichment / pruning) ---
Write-Host ""
Write-Host "[Step 2/3] pyscenic ctx ..."
$ctx_start = Get-Date

pyscenic ctx `
    $GRN_ADJ `
    $RANKING_DB `
    --annotations_fname $MOTIF_TBL `
    --expression_mtx_fname $EXPR_LOOM `
    --output $CTX_REG `
    --num_workers 8 `
    --mask_dropouts

$ctx_end = Get-Date
Write-Host "[Step 2/3] CTX done. Time: $(($ctx_end - $ctx_start).TotalMinutes.ToString('F1')) min"
Write-Host "  Output: $CTX_REG"

# --- Step 3: AUCell (regulon activity scoring) ---
Write-Host ""
Write-Host "[Step 3/3] pyscenic aucell ..."
$auc_start = Get-Date

pyscenic aucell `
    $EXPR_LOOM `
    $CTX_REG `
    --output $AUCELL_LOOM `
    --num_workers 8

$auc_end = Get-Date
Write-Host "[Step 3/3] AUCell done. Time: $(($auc_end - $auc_start).TotalMinutes.ToString('F1')) min"
Write-Host "  Output: $AUCELL_LOOM"

Write-Host ""
Write-Host "============================================================"
Write-Host "[E005-pyscenic] Pipeline DONE."
Write-Host "  Next: python scripts\run_e005_score_grn_space.py"
Write-Host "============================================================"
