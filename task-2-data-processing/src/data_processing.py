#!/usr/bin/env python
# coding: utf-8

## idg forked from @levanpon1009 on kaggle.com
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
import numpy as np
import os

def get_data_generators(DATA_DIR = os.getcwd()[:40] + '/data/',
                        PROJECT_NAME = 'Pneumonia-Classification',
                        BATCH_SIZE = 16, # from CheXNet paper
                        DIM = (224, 224)):

    df = pd.read_csv(DATA_DIR + 'Data_Entry_2017.csv') # TEST '/home/jinen/git/Pneumonia-Classification/data/
    print(df.head())

    data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}
    df['path'] = df['Image Index'].map(data_image_paths.get)

    labels = []
    for i in df['Finding Labels']:
        if 'Pneumonia' in i:
            labels.append(1)
        else:
            labels.append(0)
    df['Finding Labels'] = np.array(labels)
    df['path'] = df['path'].astype('str')
    train_df, valid_df = train_test_split(df, test_size=0.20, random_state=2018, stratify=df['Finding Labels'])

    core_generator = ImageDataGenerator(rescale=1 / 255,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        horizontal_flip=True,
                                        vertical_flip=False)

    training_generator = core_generator.flow_from_dataframe(dataframe=train_df,
                                                            directory=None,
                                                            x_col='path',
                                                            y_col='Finding Labels',
                                                            class_mode='raw',
                                                            batch_size=BATCH_SIZE,
                                                            target_size=DIM)

    validation_generator = core_generator.flow_from_dataframe(dataframe=valid_df,
                                                              directory=None,
                                                              x_col='path',
                                                              y_col='Finding Labels',
                                                              class_mode='raw',
                                                              batch_size=BATCH_SIZE,
                                                              target_size=DIM)

    test_X, test_Y = next(core_generator.flow_from_dataframe(dataframe=valid_df,
                                                             directory=None,
                                                             x_col='path',
                                                             y_col='Finding Labels',
                                                             class_mode='raw',
                                                             batch_size=1024,
                                                             target_size=DIM))

    return training_generator, validation_generator, test_X, test_Y
