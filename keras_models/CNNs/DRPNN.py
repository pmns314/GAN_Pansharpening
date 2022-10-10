import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, concatenate
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator

EPS = 1e-12


class DRPNN(tf.keras.Model):

    def __init__(self, channels, name="DRPNN"):
        super(DRPNN, self).__init__()
        self._model_name = name
        self.conv_pre = Conv2D(32, 7, padding="same", activation="relu", use_bias=True)
        self.rep = [Conv2D(32, 7, padding="same", activation="relu", use_bias=True) for _ in range(10)]
        self.conv_post = Conv2D(channels + 1, 7, padding="same", use_bias=True)
        self.conv_final = Conv2D(channels, 7, padding="same", use_bias=True)

    def call(self, inputs):
        try:
            pan, ms, ms_lr = inputs
        except ValueError:
            pan, ms = inputs
        inputs = concatenate([ms, pan])
        out = self.conv_pre(inputs)
        for layer in self.rep:
            out = layer(out)
        out = self.conv_post(out)
        out = self.conv_final(out)
        return out

    @property
    def name(self):
        return self._model_name


if __name__ == '__main__':

    train = True
    model = DRPNN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:

        model.compile(
            optimizer=Adam(),
            loss='mse'
        )
        # model.build([(None, None, None, 8), (None, None, None ,1)])
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

    pan, ms, _ = next(test_gen.generate_batch())[0]

    gen = model.predict([pan, ms])

    model.evaluate(test_gen)
