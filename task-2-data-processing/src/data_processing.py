#!/usr/bin/env python
# coding: utf-8

## idg forked from @levanpon1009 on kaggle.com
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from itertools import chain
from glob import glob
import pandas as pd
import numpy as np
import os


CWD = os.getcwd()
PROJECT_NAME = 'Pneumonia-Classification'
BATCH_SIZE = 16 # from CheXNet paper 
DIM = (256, 256)

def get_data_generators(CWD = os.getcwd(),
                        PROJECT_NAME = 'Pneumonia-Classification',
                        BATCH_SIZE = 16, # from CheXNet paper
                        DIM = (256, 256)):

    DATA_DIR = CWD[:40] + '/data/'

    df = pd.read_csv(DATA_DIR + 'Data_Entry_2017.csv') # TEST '/home/jinen/git/Pneumonia-Classification/data/
    df.head()

    data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}
    df['path'] = df['Image Index'].map(data_image_paths.get)


    df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    labels = [x for x in labels if len(x) > 0]
    for label in labels:
        if len(label) > 1:
            df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
    df.head()

    train_df, valid_df = train_test_split(df, test_size=0.20, random_state=2018, stratify=df['Finding Labels'].map(lambda x: x[:4]))
    train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

    core_generator = ImageDataGenerator(rescale=1 / 255,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        horizontal_flip=True,
                                        vertical_flip=False,
                                        height_shift_range=0.05,
                                        width_shift_range=0.1,
                                        rotation_range=5,
                                        shear_range=0.1,
                                        fill_mode='reflect',
                                        zoom_range=0.15)

    training_generator = core_generator.flow_from_dataframe(dataframe=train_df,
                                                    directory=None,
                                                    x_col='path',
                                                    y_col=labels,
                                                    class_mode='raw',
                                                    batch_size=BATCH_SIZE,
                                                    classes=labels,
                                                    target_size=DIM)

    validation_generator = core_generator.flow_from_dataframe(dataframe=valid_df,
                                                        directory=None,
                                                        x_col='path',
                                                        y_col=labels,
                                                        class_mode='raw',
                                                        batch_size=BATCH_SIZE,
                                                        classes=labels,
                                                        target_size=DIM)

    test_generator = core_generator.flow_from_dataframe(dataframe=valid_df,
                                                        directory=None,
                                                        x_col='path',
                                                        y_col=labels,
                                                        class_mode='raw',
                                                        batch_size=BATCH_SIZE,
                                                        classes=labels,
                                                        target_size=DIM)

    return training_generator, validation_generator, test_generator

get_data_generators()
