import os

BASE_PATH = os.getcwd()
RAW_VAL_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/val')

PROD_MODEL_PATH = os.path.join(BASE_PATH,'task-4-evaluation/model/prod_model')
PRED_PATH = os.path.join(BASE_PATH,'task-4-evaluation/eval/predictions')
HISTORY_PARAM_PATH = os.path.join(BASE_PATH,'task-4-evaluation/model/TF-Model-Checkpoint/history_params.json')
CHECKPOINT_PATH = os.path.join(BASE_PATH,'task-4-evaluation/model/TF-Model-Checkpoint/')
DEL_CHECKPOINT = False

DH_LOG_MET_PATH = os.path.join(BASE_PATH,'task-4-evaluation/eval/metrics/TF_train_final_metrics.csv')
DH_LOG_PARAM_PATH = os.path.join(BASE_PATH,'task-4-evaluation/eval/params/TF_train_final_params.yml')
