import mlflow

import os
import tensorflow as tf
from tensorflow.keras.applications import densenet

from const.modeling_const import PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, INIT_EPOCHS, LEARNING_RATE, CSV_LOG_PATH, CHECKPOINT_PATH, DH_LOG_PARAM_PATH

from task2.data_processing import get_data_generators

#from base.utiles.functions import print_data, load_dataset
from base.const.general_const import NOTEBOOK, IMG_SIZE, IMG_SHAPE, CLASS_NAME_PATH,\
    CLASS_MODE, BATCH_SIZE, PROD_MODEL_PATH, PROJECT_NAME
from dagshub import dagshub_logger

def build_model(img_shape=IMG_SHAPE,
                learning_rate=LEARNING_RATE):

    model = densenet.DenseNet121(weights='imagenet',
                             include_top=False,
                             input_shape=img_shape, pooling="avg")

    predictions = tf.keras.layers.Dense(1,
                                        activation='sigmoid',
                                        name='predictions')(model.output)
    model = tf.keras.Model(inputs=model.input, outputs=predictions)
    model.layers.pop()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def get_callbacks(csv_logger_path, checkpoint_filepath):
    csv_logger = tf.keras.callbacks.CSVLogger(csv_logger_path)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy')

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True)

    return csv_logger, early_stopping, model_checkpoint


if __name__ == '__main__':

    DAGSHUB_USER_NAME = input("DAGsHub Username:")
    DAGSHUB_TOKEN = input(f"Token for {DAGSHUB_USER_NAME}:")
    DAGSHUB_REPO_NAME = "Pneumonia-Classification"
    DAGSHUB_REPO_OWNER = "nirbarazida"

    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

    mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow')

    training_generator, validation_generator, test_X, test_Y = get_data_generators(DATA_DIR='/home/jinen/git/Pneumonia-Classification/data/',
                                                                                   PROJECT_NAME=PROJECT_NAME,
                                                                                   BATCH_SIZE=BATCH_SIZE,
                                                                                   DIM=IMG_SIZE)

    ## Temporary
    #class_names = train_dataset.class_names
    #with open(CLASS_NAME_PATH, "w") as textfile:
        #textfile.write(",".join(class_names))

    #if NOTEBOOK:
        #print_data(validation_dataset, class_names, notebook=NOTEBOOK, process=False, save=False, predict=False)

    model = build_model(IMG_SHAPE, LEARNING_RATE)
    csv_logger, early_stopping, model_checkpoint = get_callbacks(CSV_LOG_PATH, CHECKPOINT_PATH)
    history = model.fit(training_generator,
                        epochs=INIT_EPOCHS,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        callbacks=[csv_logger, early_stopping, model_checkpoint])

    with dagshub_logger(should_log_metrics=False,hparams_path=DH_LOG_PARAM_PATH) as logger:
        logger.log_hyperparams(model.history.params)

    model.save(PROD_MODEL_PATH)
