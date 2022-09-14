from devcode.models.local_learning import LocalModel
from devcode.utils import dummie2multilabel, cm2acc, scale_feat

# SETTING PATHS FOR SAVING OUTPUT RESULTS
GLOBAL_BASE_PATH = "results/code-ocean"
PRE_PROCESS_PATH = f"{GLOBAL_BASE_PATH}/pre-process-data"


GENERAL_RESULT_PATH = f"{GLOBAL_BASE_PATH}/evaluation"

GLOBAL_MODEL_RESULT_PATH   = f"{GLOBAL_BASE_PATH}/evaluation/global"
LOCAL_MODEL_RESULT_PATH    = f"{GLOBAL_BASE_PATH}/evaluation/local"
REGIONAL_MODEL_RESULT_PATH = f"{GLOBAL_BASE_PATH}/evaluation/regional"

# Pre-process optimal regions (k_opt) for local/regional models
K_OPT_PATH = f"{PRE_PROCESS_PATH}/optimal-regions"

SAVE_IMAGE_PATH = f"{GLOBAL_BASE_PATH}/images"

EXTRA_SAVE_IMAGE_PATH      = f"{SAVE_IMAGE_PATH}/extras"
CLUSTERING_SAVE_IMAGE_PATH = f"{SAVE_IMAGE_PATH}/clustering"
