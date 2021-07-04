import os

# Data path
BASE_PATH = os.getcwd()
PROCESSED_TRAIN_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/train')
PROCESSED_TEST_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/test')

# Dataset params
CLASS_MODE = 'categorical'

# Model params
INIT_EPOCHS = 1
LEARNING_RATE = 1e-6
BATCH_SIZE = 32

# Save path
CSV_LOG_PATH = os.path.join(BASE_PATH,'task-3-modeling/eval/metrics/TF_training_logs.csv')
CHECKPOINT_PATH = os.path.join(BASE_PATH,'task-3-modeling/model/TF-Model-Checkpoint/')
CLASS_NAME_PATH = os.path.join(os.getcwd(),"task-3-modeling/src/const/class_names.txt")
