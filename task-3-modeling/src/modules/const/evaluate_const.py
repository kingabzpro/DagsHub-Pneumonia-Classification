import os

BASE_PATH = os.getcwd()
RAW_VAL_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/val')

PROD_MODEL_PATH = 'model/prod_model'
PRED_PATH = 'eval/predictions'
HISTORY_PARAM_PATH = "model/TF-Model-Checkpoint/history_params.json"
CHECKPOINT_PATH = 'model/TF-Model-Checkpoint/'
DEL_CHECKPOINT = False

DH_LOG_MET_PATH = 'eval/metrics/TF_train_final_metrics.csv'
DH_LOG_PARAM_PATH = 'eval/params/TF_train_final_params.yml'
