import os
import shutil

import keras.layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, concatenate, Conv2DTranspose, ReLU
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator

EPS = 1e-12


class PanNet(tf.keras.Model):
    class ResidualBlock(keras.layers.Layer):
        def __init__(self):
            super(PanNet.ResidualBlock, self).__init__()
            self.conv1 = Conv2D(32, 3, padding='same', use_bias=True)
            self.conv2 = Conv2D(32, 3, padding='same', use_bias=True)
            self.relu = ReLU()

        def call(self, inputs, **kwargs):
            x = self.conv1(inputs)
            x = self.relu(x)
            x = self.conv2(x)
            return inputs + x

    def __init__(self, channels, name="PanNet"):
        super(PanNet, self).__init__()
        self._model_name = name
        self.deconv = Conv2DTranspose(channels, 8, strides=(4, 4), padding='same', use_bias=True)
        self.conv1 = Conv2D(32, 3, padding='same', use_bias=True)
        self.backbone = [PanNet.ResidualBlock() for _ in range(4)]
        self.conv3 = Conv2D(channels, 3, padding='same', use_bias=True)
        self.relu = ReLU()

    def call(self, inputs):
        pan, ms, ms_lr = inputs

        output_deconv = self.deconv(ms_lr)
        concat = concatenate([output_deconv, pan])
        out = self.conv1(concat)
        out = self.relu(out)
        for layer in self.backbone:
            out = layer(out)
        out = self.conv3(out)

        out = ms + out
        return out

    @property
    def name(self):
        return self._model_name

if __name__ == '__main__':

    train = True
    model = PanNet(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5", preprocessing=True)
    val_gen = DataGenerator("../../datasets/valid.h5", preprocessing=True)
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False, preprocessing=True)

    if train is True:

        if os.path.exists(path):
            shutil.rmtree(path)

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='mse'
        )
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
        loss='mse'
    )
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_ms_lr = np.random.randn(3, 32, 32, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model.evaluate([random_pan, random_ms, random_ms_lr], random_gt)

    pan, ms, ms_lr = next(test_gen.generate_batch())[0]

    gen = model.predict([pan, ms, ms_lr])
    model.evaluate(test_gen)
