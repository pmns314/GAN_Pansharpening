import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import Conv2D, LeakyReLU, concatenate, Conv2DTranspose, ReLU
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator
from keras_models.GANs.PSGAN import PSGAN

EPS = 1e-12


class FUPSGAN(PSGAN):

    def __init__(self, channels, name="FUPSGAN"):
        super(FUPSGAN, self).__init__(channels, name)

    def create_generator(self, channels):
        pan = tf.keras.layers.Input((None, None, 1))
        ms_lr = tf.keras.layers.Input((None, None, channels))

        # Pan Encoder
        pan_enc = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(pan)
        pan_enc = LeakyReLU(alpha=.2)(pan_enc)
        pan_enc_2 = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(pan_enc)
        pan_enc = LeakyReLU(alpha=.2)(pan_enc_2)
        pan_enc = Conv2D(64, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(pan_enc)

        # MS Encoder
        ms_enc = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(ms_lr)
        ms_enc = LeakyReLU(alpha=.2)(ms_enc)
        ms_enc_2 = Conv2DTranspose(32, 4, strides=(4, 4), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            ms_enc)
        ms_enc = LeakyReLU(alpha=.2)(ms_enc_2)
        ms_enc = Conv2D(64, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(ms_enc)

        # Concatenation
        conc = concatenate([pan_enc, ms_enc])

        # Encoder path
        enc = LeakyReLU(alpha=.2)(conc)
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
        dec = LeakyReLU(alpha=.2)(dec)
        dec = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(dec)

        common = concatenate([dec, enc])
        common = LeakyReLU(alpha=.2)(common)
        common = Conv2D(128, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(common)
        common = LeakyReLU(alpha=.2)(common)
        common = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            common)

        final_part = concatenate([common, pan_enc_2, ms_enc_2])

        final_part = LeakyReLU(alpha=.2)(final_part)
        final_part = Conv2D(64, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            final_part)
        final_part = LeakyReLU(alpha=.2)(final_part)
        final_part = Conv2D(channels, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            final_part)
        output = ReLU()(final_part)

        return tf.keras.models.Model([ms_lr, pan], output, name="generator")

    def loss_generator(self, ms, pan, gt, ms_lr=None):
        outputs = self.generator([ms_lr, pan])
        predict_fake = self.discriminator([ms, outputs])

        # From Code
        # gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake+EPS))
        # gen_loss_L1 = tf.reduce_mean(tf.abs(gt - outputs))
        # gen_loss = gen_loss_GAN * self.alpha + gen_loss_L1 * self.beta

        # From Formula
        GAN_loss = tf.reduce_mean(tf.math.log(predict_fake + EPS))
        loss_l1 = tf.reduce_mean(tf.math.abs(gt - outputs))
        gen_loss = - self.alpha * GAN_loss + self.beta * loss_l1

        return gen_loss

    def train_step(self, data):
        original_pans, original_ms, original_ms_lr = data[0]
        ground_truth = data[1]

        # Generate Data for Discriminators Training
        generated_HRMS = self.generator([original_ms_lr, original_pans])

        # Training Discriminator
        with tf.GradientTape() as tape:
            loss_value = self.loss_discriminator(original_ms, ground_truth, generated_HRMS)
            trainable_vars = self.discriminator.trainable_variables
            grads = tape.gradient(loss_value, trainable_vars)
        self.disc_opt.apply_gradients(zip(grads, trainable_vars))
        self.disc_loss_tracker.update_state(loss_value)  # Add current batch loss

        # Training Generator
        with tf.GradientTape() as tape:
            loss_value = self.loss_generator(original_ms, original_pans, ground_truth, original_ms_lr)
            train_vars = self.generator.trainable_variables
            grads = tape.gradient(loss_value, train_vars)
        self.gen_opt.apply_gradients(zip(grads, train_vars))
        self.gen_loss_tracker.update_state(loss_value)  # Add current batch loss

        return {"gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result()
                }

    def test_step(self, data):
        try:
            original_pans, original_ms, original_ms_lr = data[0]
            ground_truth = data[1]
        except ValueError:
            original_pans, original_ms, ground_truth, original_ms_lr = data[0]

        generated_HRMS = self.generator([original_ms_lr, original_pans])
        self.disc_loss_tracker.update_state(self.loss_discriminator(original_ms, ground_truth, generated_HRMS))
        self.gen_loss_tracker.update_state(
            self.loss_generator(original_ms, original_pans, ground_truth, original_ms_lr))

        return {"gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result()
                }


if __name__ == '__main__':

    train = True
    model = FUPSGAN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:

        if os.path.exists(path):
            shutil.rmtree(path)

        model.generator.summary()
        model.discriminator.summary()

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

    model2 = FUPSGAN(8)
    model2.load_weights(f'{path}\\weights')
    model2.compile()
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    random_ms_lr = np.random.randn(3, 32, 32, 8)
    model2.evaluate([random_pan, random_ms, random_gt, random_ms_lr])

    a, b, d = next(test_gen.generate_batch())[0]

    gen = model2.generator([d, a])
