import os
import shutil

import keras.layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, concatenate, ReLU
from keras.optimizers import Adam, SGD

from dataset.DataGeneratorKeras import DataGenerator


class APNN(tf.keras.Model):

    def __init__(self, channels, name="APNN"):
        super(APNN, self).__init__()
        self._model_name = name
        self.channels = channels
        self.conv1 = Conv2D(64, 9, padding='same', use_bias=True, activation='relu')
        self.conv2 = Conv2D(32, 5, padding='same', use_bias=True, activation='relu')
        self.conv3 = Conv2D(channels, 5, padding='same', use_bias=True)

        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        try:
            pan, ms, _ = inputs
        except ValueError:
            pan, ms = inputs

        inputs = concatenate([ms, pan])
        rs = self.conv1(inputs)
        rs = self.conv2(rs)
        out = self.conv3(rs)

        # Skip connection converts the model in a residual model
        out = ms + out
        return out

    @property
    def name(self):
        return self._model_name


if __name__ == '__main__':

    train = True
    model = APNN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:
        model.compile(
            optimizer=Adam(),
            loss='mse'
        )

        if os.path.exists(path):
            shutil.rmtree(path)

        cb = [
            EarlyStopping(patience=15, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(f'{path}\\weights',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True),
            TensorBoard(log_dir=f'{path}\\log')
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=1, callbacks=cb)

    model.load_weights(f'{path}\\weights')
    model.compile(
        optimizer=Adam(),
        loss='mae'
    )

    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model.evaluate([random_pan, random_ms], random_gt)

    pan, ms, _ = next(test_gen.generate_batch())[0]

    gen = model.predict([pan, ms])

    model.evaluate(test_gen)
