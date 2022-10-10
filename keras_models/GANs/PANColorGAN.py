import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, LeakyReLU, concatenate, Conv2DTranspose, Input, BatchNormalization, Dropout
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator


def _add_conv(input_tensor, n_weights, kernel_size, stride=1, use_dropout=False):
    layer = Conv2D(n_weights, kernel_size, padding='same', strides=(stride, stride))(input_tensor)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=.2)(layer)
    if use_dropout:
        layer = Dropout(.2)(layer)
    return layer


def _add_transpose(input_tensor, n_weights, kernel_size, stride=1):
    layer = Conv2DTranspose(n_weights, kernel_size, padding='same', strides=(stride, stride))(input_tensor)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=.2)(layer)
    return layer


def create_discriminator(channels):
    inputs = Input((None, None, 2 * channels + 1))

    _ = Conv2D(32, 4, strides=(2, 2), padding='same')(inputs)
    _ = LeakyReLU(alpha=.2)(_)
    _ = _add_conv(_, 64, 4, stride=2)
    _ = _add_conv(_, 128, 4, stride=2)
    _ = _add_conv(_, 256, 4, stride=2)
    _ = _add_conv(_, 256, 4)

    output = Conv2D(1, 4, padding='same', activation='sigmoid')(_)
    return tf.keras.models.Model([inputs], output, name="discriminator")


def create_generator(channels):
    input_ms = Input((None, None, channels))
    input_gray = Input((None, None, 1))

    # Color Injection
    color1 = _add_conv(input_ms, 32, 3)
    color1 = _add_conv(color1, 32, 3)
    color2 = _add_conv(color1, 64, 3, stride=2)
    color3 = _add_conv(color2, 128, 3, stride=2)

    # Spatial Detail Extraction
    model1 = _add_conv(input_gray, 32, 3)
    model1 = _add_conv(model1, 32, 3)
    mc1 = concatenate([model1, color1])
    model2 = _add_conv(mc1, 64, 3, stride=2)
    mc2 = concatenate([model2, color2])
    model3 = _add_conv(mc2, 128, 3, stride=2)
    mc3 = concatenate([model3, color3])

    # Feature Transformation
    resnet_block = None
    input_resnet_block = mc3
    for i in range(6):
        rb = _add_conv(input_resnet_block, 256, 3, use_dropout=True)
        rb = _add_conv(rb, 256, 3)
        resnet_block = input_resnet_block + rb
        input_resnet_block = resnet_block

    # Image Synthesis
    model4 = _add_conv(resnet_block, 128, 3)
    m34 = concatenate([model3, model4])
    model5 = _add_transpose(m34, 128, 3, stride=2)
    model6 = _add_conv(model5, 64, 3)
    m26 = concatenate([model2, model6])
    model7 = _add_transpose(m26, 64, 3, stride=2)
    model8 = _add_conv(model7, 32, 3)
    m18 = concatenate([model1, model8])
    model9 = _add_conv(m18, 32, 3)

    output = Conv2D(channels, 3, padding='same', activation='tanh')(model9)

    return tf.keras.models.Model([input_ms, input_gray], output, name="generator")


class PANColorGAN(tf.keras.Model):

    def __init__(self, channels, name="PANColorGAN"):
        super(PANColorGAN, self).__init__()
        self._model_name = name
        self.generator = create_generator(channels)
        self.discriminator = create_discriminator(channels)

        self.gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')

        self.gen_opt = None
        self.disc_opt = None

    @property
    def name(self):
        return self._model_name

    def loss_discriminator(self, ms, pan, gt, generated):
        fake_ab = concatenate([ms, pan, generated])
        real_ab = concatenate([ms, pan, gt])

        pred_fake = self.discriminator(fake_ab)
        pred_real = self.discriminator(real_ab)

        ones = tf.ones_like(pred_fake)
        zeros = tf.zeros_like(pred_fake)
        mse = tf.keras.losses.MeanSquaredError()

        # Label 1 for fake, Label 0 for True
        loss_d_fake = mse(pred_fake - tf.reduce_mean(pred_real), ones)
        loss_d_real = mse(pred_real - tf.reduce_mean(pred_fake), zeros)

        return (loss_d_real + loss_d_fake) / 2

    def loss_generator(self, ms, pan, gt):
        generated = self.generator([ms, pan])

        fake_ab = concatenate([ms, pan, generated])
        real_ab = concatenate([ms, pan, gt])

        pred_fake = self.discriminator(fake_ab)
        pred_real = self.discriminator(real_ab)

        ones = tf.ones_like(pred_fake)
        zeros = tf.zeros_like(pred_fake)
        mse = tf.keras.losses.MeanSquaredError()

        loss_g_real = mse(pred_real - tf.reduce_mean(pred_fake), zeros)
        loss_g_fake = mse(pred_fake - tf.reduce_mean(pred_real), ones)

        return (loss_g_real + loss_g_fake) / 2

    def train_step(self, data):
        original_pans, original_ms, _ = data[0]
        ground_truth = data[1]

        # Generate Data for Discriminator Training
        generated_HRMS = self.generator([original_ms, original_pans])

        # Training Discriminator
        with tf.GradientTape() as tape:
            loss_value = self.loss_discriminator(original_ms, original_pans, ground_truth, generated_HRMS)
            trainable_vars = self.discriminator.trainable_variables
            grads = tape.gradient(loss_value, trainable_vars)
        self.disc_opt.apply_gradients(zip(grads, trainable_vars))
        self.disc_loss_tracker.update_state(loss_value)  # Add current batch loss

        # Training Generator
        with tf.GradientTape() as tape:
            loss_value = self.loss_generator(original_ms, original_pans, ground_truth)
            train_vars = self.generator.trainable_variables
            grads = tape.gradient(loss_value, train_vars)
        self.gen_opt.apply_gradients(zip(grads, train_vars))
        self.gen_loss_tracker.update_state(loss_value)  # Add current batch loss

        return {"gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result()
                }

    def test_step(self, data):
        try:
            original_pans, original_ms, _ = data[0]
            ground_truth = data[1]
        except ValueError:
            original_pans, original_ms, ground_truth = data[0]
        generated_HRMS = self.generator([original_ms, original_pans])
        self.disc_loss_tracker.update_state(
            self.loss_discriminator(original_ms, original_pans, ground_truth, generated_HRMS))
        self.gen_loss_tracker.update_state(self.loss_generator(original_ms, original_pans, ground_truth))

        return {"gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result()
                }

    def compile(self, gen_optimizer=None, disc_optimizer=None):
        super(PANColorGAN, self).compile()
        self.gen_opt = gen_optimizer
        self.disc_opt = disc_optimizer

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def reset_metrics(self):
        self.gen_loss_tracker.reset_states()
        self.disc_loss_tracker.reset_states()


if __name__ == '__main__':
    name_model = "PANColorGAN"
    train = True
    path = f"..\\trained_models\\{name_model}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:
        path = f"..\\trained_models\\{name_model}_model"
        if os.path.exists(path):
            shutil.rmtree(path)

        model = PANColorGAN(8)
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

    model2 = PANColorGAN(8)
    model2.load_weights(f'..\\trained_models\\{name_model}_model\\weights')
    model2.compile()
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model2.evaluate([random_pan, random_ms, random_gt])

    pan, ms, _ = next(test_gen.generate_batch())[0]

    gen = model2.generator([ms, pan])
