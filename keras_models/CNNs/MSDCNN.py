import os
import shutil

import keras.layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, concatenate, ReLU
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator


class MSDCNN(tf.keras.Model):
    class MSResB(keras.layers.Layer):
        def __init__(self, weights):
            super(MSDCNN.MSResB, self).__init__()
            self.conv1 = Conv2D(weights, 3, padding='same', use_bias=True, activation='relu')
            self.conv2 = Conv2D(weights, 5, padding='same', use_bias=True, activation='relu')
            self.conv3 = Conv2D(weights, 7, padding='same', use_bias=True, activation='relu')
            self.relu = ReLU()

        def call(self, inputs, **kwargs):
            x = self.conv1(inputs)
            y = self.conv2(inputs)
            z = self.conv3(inputs)

            concat = concatenate([x, y, z])
            out = concat + inputs
            out = self.relu(out)
            return out

    def __init__(self, channels, name="MSDCNN"):
        super(MSDCNN, self).__init__()
        self._model_name = name
        self.conv_1_1 = Conv2D(60, 7, padding='same', use_bias=True, activation='relu')
        self.msresb_1_1 = MSDCNN.MSResB(20)
        self.conv_1_2 = Conv2D(30, 3, padding='same', use_bias=True, activation='relu')
        self.msresb_1_2 = MSDCNN.MSResB(10)
        self.conv_1_3 = Conv2D(channels, 5, padding='same', use_bias=True)

        self.conv_2_1 = Conv2D(64, 9, padding='same', use_bias=True, activation='relu')
        self.conv_2_2 = Conv2D(32, 1, padding='same', use_bias=True, activation='relu')
        self.conv_2_3 = Conv2D(channels, 5, padding='same', use_bias=True)

        self.relu = ReLU()

    def call(self, inputs):
        try:
            pan, ms, _ = inputs
        except ValueError:
            pan, ms = inputs
        inputs = concatenate([ms, pan])

        # Deep Features
        deep_f = self.conv_1_1(inputs)
        deep_f = self.msresb_1_1(deep_f)
        deep_f = self.conv_1_2(deep_f)
        deep_f = self.msresb_1_2(deep_f)
        deep_f = self.conv_1_3(deep_f)

        # Shallow Features
        shallow_f = self.conv_2_1(inputs)
        shallow_f = self.conv_2_2(shallow_f)
        shallow_f = self.conv_2_3(shallow_f)

        out = deep_f + shallow_f
        out = self.relu(out)

        return out

    @property
    def name(self):
        return self._model_name


if __name__ == '__main__':

    train = True
    model = MSDCNN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:

        model.compile(
            optimizer=Adam(),
            loss='mse'
        )
        model.build([(None, 64, 64, 8), (None, 64, 64, 1)])
        model.summary()

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

    pan, ms, _ = next(test_gen.generate_batch())[0]

    gen = model.predict([pan, ms])

    model.evaluate(test_gen)
