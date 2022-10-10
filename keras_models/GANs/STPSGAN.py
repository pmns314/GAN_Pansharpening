import os
import shutil
import tensorflow as tf
import numpy as np
from keras.initializers.initializers_v1 import RandomNormal
from keras.optimizers import Adam
from keras.layers import Conv2D, LeakyReLU, concatenate, Conv2DTranspose, ReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from dataset.DataGeneratorKeras import DataGenerator
from keras_models.GANs.PSGAN import PSGAN


class STPSGAN(PSGAN):

    def __init__(self, channels, name="STPSGAN"):
        super().__init__(channels, name)

    def create_generator(self, channels):
        pan = tf.keras.layers.Input((None, None, 1))
        ms = tf.keras.layers.Input((None, None, channels))

        # Concatenation
        inputs = concatenate([pan, ms])

        main_stream = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(inputs)
        main_stream = LeakyReLU(alpha=.2)(main_stream)
        main_stream = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            main_stream)

        # Encoder path
        enc = LeakyReLU(alpha=.2)(main_stream)
        enc = Conv2D(64, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(enc)
        enc = LeakyReLU(alpha=.2)(enc)
        enc = Conv2D(128, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(enc)
        enc = LeakyReLU(alpha=.2)(enc)
        enc = Conv2D(128, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(enc)

        # Decoder Path
        dec = LeakyReLU(alpha=.2)(enc)
        dec = Conv2D(256, 3, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(dec)
        dec = LeakyReLU(alpha=.2)(dec)
        dec = Conv2D(256, 1, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(dec)
        dec = LeakyReLU(alpha=.2)(dec)
        dec = Conv2D(256, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(dec)

        dec2 = LeakyReLU(alpha=.2)(dec)
        dec2 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            dec2)

        common = concatenate([dec2, enc])

        common = LeakyReLU(alpha=.2)(common)
        common = Conv2D(128, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(common)
        common = LeakyReLU(alpha=.2)(common)
        common = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            common)

        final_part = concatenate([common, main_stream])

        final_part = LeakyReLU(alpha=.2)(final_part)
        final_part = Conv2D(64, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            final_part)
        final_part = LeakyReLU(alpha=.2)(final_part)
        final_part = Conv2D(channels, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            final_part)
        output = ReLU()(final_part)

        return tf.keras.models.Model([ms, pan], output, name="generator")


if __name__ == '__main__':

    train = True
    model = STPSGAN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:

        if os.path.exists(path):
            shutil.rmtree(path)

        # model.generator.summary()
        # model.discriminator.summary()

        model.compile(
            gen_optimizer=Adam(),
            disc_optimizer=Adam()
        )
        cb = [
            EarlyStopping(monitor="val_gen_loss", patience=15, verbose=1),
            ReduceLROnPlateau(monitor="val_gen_loss", factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(f'{path}\\weights',
                            monitor="val_gen_loss",
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True),
            TensorBoard(log_dir=f'{path}\\log')
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=1, callbacks=cb)

    model.load_weights(f'{model.name}\\weights')
    model.compile()
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model.evaluate([random_pan, random_ms, random_gt])

    pan, ms, _ = next(test_gen.generate_batch())[0]

    gen = model.generator([ms, pan])
