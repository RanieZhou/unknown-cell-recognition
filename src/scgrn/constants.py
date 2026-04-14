"""Project-wide constants for the refactored scGRN mainline."""

METHOD_NAME = "lineage_selective_rescue_globalT"
DEFAULT_BACKBONE_NAME = "scanvi"
SUPPORTED_BACKBONES = ("scanvi", "scnym")
EXPRESSION_SCHEMA_VERSION = "backbone_expression_v1"
SPLIT_COLUMN = "split"
LEGACY_SPLIT_COLUMN = "E005" "_split"

SPLIT_TRAIN = "train_known"
SPLIT_VAL = "val_known"
SPLIT_TEST_KNOWN = "test_known"
SPLIT_TEST_UNKNOWN = "test_unknown"
TEST_SPLITS = [SPLIT_TEST_KNOWN, SPLIT_TEST_UNKNOWN]

DEFAULT_LINEAGE_BUCKET_SUFFIX = "_like"

DEFAULT_METHOD_ORDER = [
    "expr_fused",
    "selective_fused_score",
    "lineage_selective_rescue_globalT",
    "grn_distance",
    "grn_aux_score",
    "dual_fused_v1",
    "dual_fused_v2_rankavg",
    "v3_ml_anomaly",
]
