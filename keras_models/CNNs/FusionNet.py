import os
import shutil

import keras.layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, ReLU
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator


class FusionNet(tf.keras.Model):
    class ResidualBlock(keras.layers.Layer):
        def __init__(self):
            super(FusionNet.ResidualBlock, self).__init__()
            self.conv1 = Conv2D(32, 3, padding='same', use_bias=True, activation='relu')
            self.conv2 = Conv2D(32, 3, padding='same', use_bias=True, activation='relu')
            self.relu = ReLU()

        def call(self, inputs, **kwargs):
            x = self.conv1(inputs)
            x = self.conv2(x)

            out = x + inputs
            out = self.relu(out)
            return out

    def __init__(self, channels, name="FusionNet"):
        super(FusionNet, self).__init__()
        self._model_name = name
        self.channels = channels
        self.conv1 = Conv2D(32, 3, padding='same', use_bias=True)
        self.backbone = [FusionNet.ResidualBlock() for _ in range(4)]
        self.conv3 = Conv2D(channels, 3, padding='same', use_bias=True)

        self.relu = ReLU()

    def call(self, inputs):
        try:
            pan, ms, _ = inputs
        except ValueError:
            pan, ms = inputs

        pan_rep = tf.repeat(pan, self.channels, axis=3)
        rs = pan_rep - ms
        rs = self.conv1(rs)
        rs = self.relu(rs)
        for layer in self.backbone:
            rs = layer(rs)
        rs = self.conv3(rs)

        out = rs + ms
        return out

    @property
    def name(self):
        return self._model_name


if __name__ == '__main__':

    train = True
    model = FusionNet(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:
        model.compile(
            optimizer=Adam(),
            loss='mse'
        )
        # model.build([(None, 64, 64, 8), (None, 64, 64, 1)])
        # model.summary()

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
        loss='mse'
    )
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model.evaluate([random_pan, random_ms], random_gt)

    pan_t, ms_t, _ = next(test_gen.generate_batch())[0]

    gen = model.predict([pan_t, ms_t])

    model.evaluate(test_gen)
