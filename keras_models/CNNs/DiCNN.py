import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, concatenate
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator

EPS = 1e-12


class DiCNN(tf.keras.Model):

    def __init__(self, channels, name="DiCNN"):
        super(DiCNN, self).__init__()
        self._model_name = name
        self.conv1 = Conv2D(64, 3, padding='same', use_bias=True, activation='relu')
        self.conv2 = Conv2D(64, 3, padding='same', use_bias=True, activation='relu')
        self.conv3 = Conv2D(channels, 3, padding='same', use_bias=True)

    def call(self, inputs):
        try:
            pan, ms, _ = inputs
        except ValueError:
            pan, ms = inputs
        inputs = concatenate([ms, pan])

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = ms + out
        return out

    @property
    def name(self):
        return self._model_name


def frobenius_loss(y_true, y_pred):
    tensor = y_pred - y_true
    norm = tf.norm(tensor, ord='fro', axis=(1, 2))
    return tf.reduce_mean(tf.square(norm))


if __name__ == '__main__':

    train = True
    model = DiCNN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:

        model.compile(
            optimizer=Adam(),
            loss=frobenius_loss,
            metrics='mse'
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
        optimizer=Adam()
    )
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model.evaluate([random_pan, random_ms], random_gt)

    pan, ms, _ = next(test_gen.generate_batch())[0]

    gen = model.predict([pan, ms])

    model.evaluate(test_gen)
