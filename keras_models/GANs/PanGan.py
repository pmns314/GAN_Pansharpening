import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import concatenate, LeakyReLU
from keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator


class PanGan(tf.keras.Model):
    def __init__(self, channels, name="PanGan"):
        super(PanGan, self).__init__()
        self._model_name = name
        self.a = .2
        self.b = .8
        self.c = .9
        self.d = .9

        self.alpha = .002
        self.beta = .001
        self.mu = 5

        self.generator = self.create_generator(channels)
        self.spatial_discriminator = self.create_discriminator(1)
        self.spectral_discriminator = self.create_discriminator(channels)

        self.epoch_spatial_loss_avg = tf.keras.metrics.Mean(name='loss_spat')
        self.epoch_spectral_loss_avg = tf.keras.metrics.Mean(name='loss_spec')
        self.epoch_generator_loss_avg = tf.keras.metrics.Mean(name='loss_gen')

        self.spat_opt = Adam()
        self.spect_opt = Adam()
        self.gen_opt = Adam()

    @property
    def name(self):
        return self._model_name

    def create_generator(self, channels):

        ms = tf.keras.layers.Input((None, None, channels))
        pan = tf.keras.layers.Input((None, None, 1))

        inputs_layer1 = concatenate([pan, ms])

        layer1 = tf.keras.layers.Conv2D(64, 9, padding="same", use_bias=True, activation="relu",
                                        kernel_initializer=RandomNormal(stddev=1e-3))(inputs_layer1)
        layer1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=.9)(layer1)

        input_layer2 = concatenate([ms, layer1, pan])
        layer2 = tf.keras.layers.Conv2D(32, 5, padding="same", use_bias=True, activation="relu",
                                        kernel_initializer=RandomNormal(stddev=1e-3))(input_layer2)
        layer2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=.9)(layer2)

        input_layer3 = concatenate([inputs_layer1, layer1, layer2])
        layer3 = tf.keras.layers.Conv2D(channels, 5, padding="same", use_bias=True, activation="tanh",
                                        kernel_initializer=RandomNormal(stddev=1e-3))(input_layer3)
        layer3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=.9)(layer3)

        return tf.keras.models.Model([ms, pan], layer3)

    def create_discriminator(self, channels):
        input_image = tf.keras.layers.Input((None, None, channels))
        _ = input_image
        for weights in [16, 32, 64, 128, 256]:
            _ = tf.keras.layers.Conv2D(weights, 3, padding="same", strides=(2, 2), use_bias=True,
                                       kernel_initializer=RandomNormal(stddev=1e-3))(_)
            if weights != 16:  # First layer doesn't have to be normalized
                _ = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=.9)(_)
            _ = LeakyReLU(alpha=.2)(_)

        out = tf.keras.layers.Conv2D(1, 2, padding="valid", use_bias=True,
                                     kernel_initializer=RandomNormal(stddev=1e-3))(_)
        out = LeakyReLU(alpha=.2)(out)

        return tf.keras.models.Model(input_image, out)

    def generator_loss(self, ms, pan):
        # downsampled = downsample(generated, ms.shape)

        generated = self.generator([ms, pan])
        averaged = tf.reduce_mean(generated, 3, keepdims=True)

        # Spectral Loss
        # L_spectral_base = tf.reduce_mean(tf.square(linalg.norm((generated - ms), 'fro')))
        L_spectral_base = tf.reduce_mean(tf.square((generated - ms)))
        L_adv1 = tf.reduce_mean(tf.square(self.spectral_discriminator(generated) - self.c))
        L_spectral = L_spectral_base + self.alpha * L_adv1

        # Spatial Loss
        details_generated = high_pass(averaged)
        details_original = high_pass(pan)
        # L_spatial_base = self.mu * tf.reduce_mean(tf.square(linalg.norm((details_generated - details_original),
        # 'fro')))
        L_spatial_base = self.mu * tf.reduce_mean(tf.square((details_generated - details_original)))
        L_adv2 = tf.reduce_mean(tf.square(self.spatial_discriminator(averaged) - self.d))
        L_spatial = L_spatial_base + self.beta * L_adv2

        return L_spatial + L_spectral

    def discriminator_spatial_loss(self, pan, generated):
        averaged = tf.reduce_mean(generated, 3, keepdims=True)
        first = tf.reduce_mean(tf.square(self.spatial_discriminator(pan) - self.b))
        second = tf.reduce_mean(tf.square(self.spatial_discriminator(averaged) - self.a))
        return first + second

    def discriminator_spectral_loss(self, ms, generated):
        first = tf.reduce_mean(tf.square(self.spectral_discriminator(ms) - self.b))
        second = tf.reduce_mean(tf.square(self.spectral_discriminator(generated) - self.a))
        return first + second

    def train_step(self, data):
        original_pans, original_ms, _ = data[0]

        # Generate Data for Discriminators Training
        generated_HRMS = self.generator([original_ms, original_pans])

        # Training Discriminators
        with tf.GradientTape() as tape:
            loss_value = self.discriminator_spatial_loss(original_pans, generated_HRMS)
            trainable_vars = self.spatial_discriminator.trainable_variables
            grads = tape.gradient(loss_value, trainable_vars)
        self.spat_opt.apply_gradients(zip(grads, trainable_vars))
        self.epoch_spatial_loss_avg.update_state(loss_value)  # Add current batch loss

        with tf.GradientTape() as tape:
            loss_value = self.discriminator_spectral_loss(original_ms, generated_HRMS)
            grads = tape.gradient(loss_value, self.spectral_discriminator.trainable_variables)
        self.spect_opt.apply_gradients(zip(grads, self.spectral_discriminator.trainable_variables))
        self.epoch_spectral_loss_avg.update_state(loss_value)  # Add current batch loss

        # Training Generator
        with tf.GradientTape() as tape:
            loss_value = self.generator_loss(original_ms, original_pans)
            grads = tape.gradient(loss_value, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.epoch_generator_loss_avg.update_state(loss_value)  # Add current batch loss

        return {"loss_spat": self.epoch_spatial_loss_avg.result(),
                "loss_spec": self.epoch_spectral_loss_avg.result(),
                "loss_gen": self.epoch_generator_loss_avg.result()
                }

    def test_step(self, data):
        try:
            original_pans, original_ms, _ = data[0]
        except:
            original_pans, original_ms = data[0]
        generated_HRMS = self.generator([original_ms, original_pans])
        self.epoch_spatial_loss_avg.update_state(self.discriminator_spatial_loss(original_pans, generated_HRMS))
        self.epoch_spectral_loss_avg.update_state(self.discriminator_spectral_loss(original_ms, generated_HRMS))
        self.epoch_generator_loss_avg.update_state(self.generator_loss(original_ms, original_pans))

        return {"loss_spat": self.epoch_spatial_loss_avg.result(),
                "loss_spec": self.epoch_spectral_loss_avg.result(),
                "loss_gen": self.epoch_generator_loss_avg.result()
                }

    def compile(self, spat_optimizer, spec_optimizer, gen_optimizer):
        super(PanGan, self).compile()
        self.spat_opt = spat_optimizer
        self.spect_opt = spec_optimizer
        self.gen_opt = gen_optimizer

    @property
    def metrics(self):
        return [self.epoch_spatial_loss_avg, self.epoch_spectral_loss_avg, self.epoch_generator_loss_avg]

    def reset_metrics(self):
        self.epoch_spatial_loss_avg.reset_states()
        self.epoch_spectral_loss_avg.reset_states()
        self.epoch_generator_loss_avg.reset_states()

    # def save_model(self, path):
    #     self.generator.save(path + "/gen")
    #     self.spatial_discriminator.save(path + "/spat_disc")
    #     self.spectral_discriminator.save(path + "/spec_disc")
    #
    # def load_model(self, path):
    #     self.generator = keras.models.load_model(path + "/gen")
    #     self.spatial_discriminator = keras.models.load_model(path + "/spat_disc")
    #     self.spectral_discriminator = keras.models.load_model(path + "/spec_disc")


# def spatial_loss(pan, generated):
#     spatial_pos = disc.predict(pan)
#     spatial_neg = disc.predict(pan)
#
#     # loss = 1/N * (predicted - 1)^2
#     spatial_pos_loss = tf.reduce_mean(
#         tf.square(spatial_pos - tf.ones(shape=[batch_size, 1], dtype=tf.float32)))
#     spatial_neg_loss = tf.reduce_mean(
#         tf.square(spatial_neg - tf.zeros(shape=[batch_size, 1], dtype=tf.float32)))
#
#     return spatial_pos_loss + spatial_neg_loss
#
#
# def spectral_loss(y_true, y_pred):
#     spectrum_pos = disc.predict(ms)
#     spectrum_neg = disc.predict(ms)
#
#     # loss = 1/N * (predicted - 1)^2
#     spectrum_pos_loss = tf.reduce_mean(
#         tf.square(spectrum_pos - tf.ones(shape=[batch_size, 1], dtype=tf.float32)))
#     spectrum_neg_loss = tf.reduce_mean(
#         tf.square(spectrum_neg - tf.zeros(shape=[batch_size, 1], dtype=tf.float32)))
#
#     return spectrum_pos_loss + spectrum_neg_loss
#
# def generator_loss(y_true, y_pred):
#     # Spatial adversarial loss = 1/N (
#     spatial_loss_ad = tf.reduce_mean(
#         tf.square(spatial_neg - tf.ones(shape=[batch_size, 1], dtype=tf.float32)))
#     tf.summary.scalar('spatial_loss_ad', spatial_loss_ad)
#
#
#     spectrum_loss_ad = tf.reduce_mean(
#         tf.square(spectrum_neg - tf.ones(shape=[batch_size, 1], dtype=tf.float32)))
#     tf.summary.scalar('spectrum_loss_ad', spectrum_loss_ad)
#
#     g_spatital_loss = tf.reduce_mean(tf.square(self.PanSharpening_img_hp - self.pan_img_hp))
#
#     tf.summary.scalar('g_spatital_loss', g_spatital_loss)
#     g_spectrum_loss = tf.reduce_mean(
#         tf.square(self.PanSharpening_img - self.ms_img_))
#     tf.summary.scalar('g_spectrum_loss', g_spectrum_loss)
#     self.g_loss = 5 * spatial_loss_ad + spectrum_loss_ad + 5 * g_spatital_loss + g_spectrum_loss
#
#     tf.summary.scalar('g_loss', self.g_loss)


def high_pass(img):
    blur_kerel = np.zeros(shape=(3, 3, 1, 1), dtype=np.float32)
    value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    blur_kerel[:, :, 0, 0] = value
    img_hp = tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1, 1, 1, 1], padding='SAME')
    return img_hp


def downsample(img, shape):
    return np.resize(img, shape)


if __name__ == '__main__':
    name_model = "PanGan"
    train = False
    path = f"..\\trained_models\\{name_model}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:
        path = f"..\\trained_models\\{name_model}_model"
        if os.path.exists(path):
            shutil.rmtree(path)

        model = PanGan(8)
        model.generator.summary()
        model.spatial_discriminator.summary()
        model.spectral_discriminator.summary()

        model.compile(
            gen_optimizer=Adam(),
            spec_optimizer=Adam(),
            spat_optimizer=Adam()
        )
        cb = [
            EarlyStopping(monitor="val_loss_gen", patience=15, verbose=1),
            ReduceLROnPlateau(monitor="val_loss_gen", factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(f'{path}\\weights',
                            monitor="val_loss_gen",
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True),
            TensorBoard(log_dir=f'{path}\\log')
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=1, callbacks=cb)

    model2 = PanGan(8)
    model2.load_weights(f'..\\trained_models\\{name_model}_model\\weights')

    model2.compile(
        gen_optimizer=Adam(),
        spec_optimizer=Adam(),
        spat_optimizer=Adam()
    )

    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    model2.evaluate([random_pan, random_ms])

    a, b, d = next(test_gen.generate_batch())[0]

    gen = model2.generator([a, b])
