import os
import shutil

import keras.layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, concatenate, ReLU

from dataset.DataGeneratorKeras import DataGenerator
from hello import ROOT_DIR


class PNN(tf.keras.Model):

    def __init__(self, channels, name="PNN"):
        super(PNN, self).__init__()
        self._model_name = name
        self.channels = channels
        self.conv1 = Conv2D(64, 9, padding='same', use_bias=True)
        self.conv2 = Conv2D(32, 5, padding='same', use_bias=True)
        self.conv3 = Conv2D(channels, 5, padding='same', use_bias=True)

        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        try:
            pan, ms, _ = inputs
        except ValueError:
            pan, ms = inputs

        inputs = concatenate([ms, pan])
        rs = self.conv1(inputs)
        rs = self.relu(rs)
        rs = self.conv2(rs)
        rs = self.relu(rs)
        out = self.conv3(rs)
        return out

    @property
    def name(self):
        return self._model_name


if __name__ == '__main__':

    train = True
    satellite = "W3"
    dataset_path = f"{ROOT_DIR}/datasets/"
    train_gen = DataGenerator(dataset_path + satellite + "/train.h5")
    val_gen = DataGenerator(dataset_path + satellite + "/val.h5")
    test_gen = DataGenerator(dataset_path + satellite + "/test.h5", shuffle=False)

    model = PNN(train_gen.channels)
    file_name = "prova"
    path = os.path.join(ROOT_DIR, 'keras_models', 'trained_models', file_name)

    if train is True:

        model.compile(
            optimizer='adam',
            loss='mse'
        )

        if os.path.exists(path):
            shutil.rmtree(path)

        cb = [
            EarlyStopping(monitor='loss', patience=15, verbose=1),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(f'{path}',
                            monitor='loss',
                            verbose=1,
                            save_best_only=True),
            TensorBoard(log_dir=f'{path}\\log')
        ]

        model.fit(train_gen, epochs=1, callbacks=cb)
