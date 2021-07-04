import json
import shutil

import tensorflow as tf
from dagshub import DAGsHubLogger

from src.const.general_const import *
from .const.evaluate_const import *
from .utiles.functions import print_data, load_dataset


def log_experiment(params, trained_model_loss, trained_model_accuracy):
    logger = DAGsHubLogger(metrics_path=DH_LOG_MET_PATH,
                           hparams_path=DH_LOG_PARAM_PATH)
    logger.log_hyperparams(params)
    logger.log_metrics(test_set_loss=trained_model_loss, test_set_accuracy=trained_model_accuracy)
    logger.save()
    logger.close()

if __name__ == '__main__':
    # Load dataset
    test_dataset = load_dataset(RAW_VAL_PATH, BATCH_SIZE, IMG_SIZE, LABEL_MODE)

    # Load production model and trained model
    prod_model = tf.keras.models.load_model(PROD_MODEL_PATH)
    trained_model = tf.keras.models.load_model(CHECKPOINT_PATH)

    # Evaluate models & compare results
    _, prod_model_accuracy = prod_model.evaluate(test_dataset)
    trained_model_loss, trained_model_accuracy = trained_model.evaluate(test_dataset)

    if trained_model_accuracy > prod_model_accuracy:
        print(f"The new model produced better results with {round(trained_model_accuracy, 2)}\
            accuracy compared to {round(prod_model_accuracy, 2)}")

        # Save model
        trained_model.save(PROD_MODEL_PATH)

        # Save new predictions
        with open(CLASS_NAME_PATH, "r") as textfile:
            class_names = textfile.read().split(',')

        print_data(test_dataset, class_names, notebook=NOTEBOOK, process=False, save=True,
                   predict=True, save_path=PRED_PATH, model=trained_model)

        log_experiment(json.load(open(HISTORY_PARAM_PATH, 'r')),
                       trained_model_loss, trained_model_accuracy)

    if DEL_CHECKPOINT:
        # Delete training checkpoint
        BASE_CHECKPOINT_PATH = os.path.join(os.getcwd(), CHECKPOINT_PATH)
        shutil.rmtree(BASE_CHECKPOINT_PATH)
