import os

BASE_PATH = os.getcwd()
RAW_TRAIN_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/train')
RAW_TEST_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/test')


INIT_EPOCHS = 1
LEARNING_RATE = 1e-6

CSV_LOG_PATH = 'eval/metrics/TF_training_logs.csv'
CHECKPOINT_PATH = 'model/TF-Model-Checkpoint/'
