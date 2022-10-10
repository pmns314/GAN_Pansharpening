import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import Conv2D, LeakyReLU, concatenate, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam

from dataset.DataGeneratorKeras import DataGenerator
from hello import ROOT_DIR

EPS = 1e-12


class PSGAN(tf.keras.Model):
    def __init__(self, channels, name="PSGAN"):
        super(PSGAN, self).__init__()
        self._model_name = name
        self.channels = channels
        self.alpha = 1
        self.beta = 100
        self.generator = self.create_generator(channels)
        self.discriminator = self.create_discriminator(channels)

        self.gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')

        self.gen_opt = None
        self.disc_opt = None

    @property
    def name(self):
        return self._model_name

    def create_generator(self, channels):
        pan = tf.keras.layers.Input((None, None, 1))
        ms = tf.keras.layers.Input((None, None, channels))
        ms_lr = tf.keras.layers.Input((None, None, channels))
        # Pan Encoder
        pan_enc = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(pan)
        pan_enc = LeakyReLU(alpha=.2)(pan_enc)
        pan_enc_2 = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(pan_enc)
        pan_enc = LeakyReLU(alpha=.2)(pan_enc_2)
        pan_enc = Conv2D(64, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(pan_enc)

        # MS Encoder
        ms_enc = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(ms)
        ms_enc = LeakyReLU(alpha=.2)(ms_enc)
        ms_enc_2 = Conv2D(32, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(ms_enc)
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

        dec2 = LeakyReLU(alpha=.2)(dec)
        dec2 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(
            dec2)

        common = concatenate([dec2, enc])

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

        return tf.keras.models.Model([pan, ms, ms_lr], output, name="generator")

    def create_discriminator(self, channels):
        inputs = tf.keras.layers.Input((None, None, channels))
        target = tf.keras.layers.Input((None, None, channels))

        input_data = concatenate([inputs, target])

        _ = Conv2D(32, 3, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(input_data)
        _ = LeakyReLU(alpha=.2)(_)
        _ = Conv2D(64, 3, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(_)
        _ = LeakyReLU(alpha=.2)(_)
        _ = Conv2D(128, 3, strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=.02))(_)
        _ = LeakyReLU(alpha=.2)(_)
        _ = Conv2D(256, 3, strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=.02))(_)
        _ = LeakyReLU(alpha=.2)(_)

        output = Conv2D(1, 3, strides=(1, 1), padding='same', activation="sigmoid",
                        kernel_initializer=RandomNormal(stddev=.02))(_)

        return tf.keras.models.Model([inputs, target], output, name="discriminator")

    def loss_generator(self, ms, pan,ms_lr, gt, *args):

        outputs = self.generator([pan, ms, ms_lr])
        predict_fake = self.discriminator([ms, outputs])
        # From Code
        # gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + EPS))
        # gen_loss_L1 = tf.reduce_mean(tf.math.abs(gt - outputs))
        # gen_loss = gen_loss_GAN * self.alpha + gen_loss_L1 * self.beta

        # From Formula
        gen_loss_GAN = tf.reduce_mean(tf.math.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.math.abs(gt - outputs))
        gen_loss = - self.alpha * gen_loss_GAN + self.beta * gen_loss_L1

        return gen_loss

    def loss_discriminator(self, ms, gt, output):

        predict_fake = self.discriminator([ms, output])
        predict_real = self.discriminator([ms, gt])

        # From Formula
        # mean[ 1 - log(fake) + log(real) ]
        return tf.reduce_mean(
            1 - tf.math.log(predict_fake + EPS) + tf.math.log(predict_real + EPS)
        )

        # From Code
        # return tf.reduce_mean(
        #     -(
        #             tf.math.log(predict_real + EPS) + tf.math.log(1 - predict_fake + EPS)
        #     )
        # )

    # def train_step(self, data):
    #     # Unpack the data.
    #     real_images, one_hot_labels = data
    #
    #     # Add dummy dimensions to the labels so that they can be concatenated with
    #     # the images. This is for the discriminator.
    #     image_one_hot_labels = one_hot_labels[:, :, None, None]
    #     image_one_hot_labels = tf.repeat(
    #         image_one_hot_labels, repeats=[image_size * image_size]
    #     )
    #     image_one_hot_labels = tf.reshape(
    #         image_one_hot_labels, (-1, image_size, image_size, num_classes)
    #     )
    #
    #     # Sample random points in the latent space and concatenate the labels.
    #     # This is for the generator.
    #     batch_size = tf.shape(real_images)[0]
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_vector_labels = tf.concat(
    #         [random_latent_vectors, one_hot_labels], axis=1
    #     )
    #
    #     # Decode the noise (guided by labels) to fake images.
    #     generated_images = self.generator(random_vector_labels)
    #
    #     # Combine them with real images. Note that we are concatenating the labels
    #     # with these images here.
    #     fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
    #     real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
    #     combined_images = tf.concat(
    #         [fake_image_and_labels, real_image_and_labels], axis=0
    #     )
    #
    #     # Assemble labels discriminating real from fake images.
    #     labels = tf.concat(
    #         [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
    #     )
    #
    #     # Train the discriminator.
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator(combined_images)
    #         d_loss = self.loss_fn(labels, predictions)
    #     grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     self.d_optimizer.apply_gradients(
    #         zip(grads, self.discriminator.trainable_weights)
    #     )
    #
    #     # Sample random points in the latent space.
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    #     random_vector_labels = tf.concat(
    #         [random_latent_vectors, one_hot_labels], axis=1
    #     )
    #
    #     # Assemble labels that say "all real images".
    #     misleading_labels = tf.zeros((batch_size, 1))
    #
    #     # Train the generator (note that we should *not* update the weights
    #     # of the discriminator)!
    #     with tf.GradientTape() as tape:
    #         fake_images = self.generator(random_vector_labels)
    #         fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
    #         predictions = self.discriminator(fake_image_and_labels)
    #         g_loss = self.loss_fn(misleading_labels, predictions)
    #     grads = tape.gradient(g_loss, self.generator.trainable_weights)
    #     self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
    #
    #     # Monitor loss.
    #     self.gen_loss_tracker.update_state(g_loss)
    #     self.disc_loss_tracker.update_state(d_loss)
    #     return {
    #         "g_loss": self.gen_loss_tracker.result(),
    #         "d_loss": self.disc_loss_tracker.result(),
    #     }

    def call(self, inputs):
         return self.generator(inputs)

    def train_step(self, data):
        original_pans, original_ms, ms_lr = data[0]
        ground_truth = data[1]

        # Generate Data for Discriminators Training
        generated_HRMS = self.generator([*data[0]])
        
        # Training Discriminator
        with tf.GradientTape() as tape:
            loss_value = self.loss_discriminator(original_ms, ground_truth, generated_HRMS)
            trainable_vars = self.discriminator.trainable_variables
            grads = tape.gradient(loss_value, trainable_vars)
        self.disc_opt.apply_gradients(zip(grads, trainable_vars))
        self.disc_loss_tracker.update_state(loss_value)  # Add current batch loss

        # Training Generator
        with tf.GradientTape() as tape:
            loss_value = self.loss_generator(original_ms, original_pans,ms_lr, ground_truth)
            train_vars = self.generator.trainable_variables
            grads = tape.gradient(loss_value, train_vars)
        self.gen_opt.apply_gradients(zip(grads, train_vars))
        self.gen_loss_tracker.update_state(loss_value)  # Add current batch loss

        return {"gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result()
                }

    def test_step(self, data):
        try:
            original_pans, original_ms, ms_lr = data[0]
            ground_truth = data[1]
        except:
            original_pans, original_ms, ground_truth = data[0]
            ms_lr = ground_truth
        generated_HRMS = self.generator([*data[0]])
        self.disc_loss_tracker.update_state(self.loss_discriminator(original_ms, ground_truth, generated_HRMS))
        self.gen_loss_tracker.update_state(self.loss_generator(original_ms, original_pans,ms_lr, ground_truth))

        return {"gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result()
                }

    def compile(self, gen_optimizer=None, disc_optimizer=None):
        super(PSGAN, self).compile()
        self.gen_opt = gen_optimizer
        self.disc_opt = disc_optimizer

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def reset_metrics(self):
        self.gen_loss_tracker.reset_states()
        self.disc_loss_tracker.reset_states()


if __name__ == '__main__':

    train = True

    satellite = "W3"
    dataset_path = f"/content/drive/MyDrive/"
    train_gen = DataGenerator(dataset_path + satellite + "/train.h5")
    val_gen = DataGenerator(dataset_path + satellite + "/val.h5")
    test_gen = DataGenerator(dataset_path + satellite + "/test.h5", shuffle=False)

    model = PSGAN(train_gen.channels)
    file_name = "PSgan_mean"
    path = os.path.join(ROOT_DIR, 'keras_models', 'trained_models', file_name)

    if train is True:

        if os.path.exists(path):
            shutil.rmtree(path)

        model.compile(
            gen_optimizer=Adam(),
            disc_optimizer=Adam()
        )
        cb = [
            EarlyStopping(monitor="val_gen_loss", patience=15, verbose=1),
            ReduceLROnPlateau(monitor="val_gen_loss", factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(f'{path}',
                            monitor="val_gen_loss",
                            verbose=1,
                            save_best_only=True),
            TensorBoard(log_dir=f'{path}\\log')
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=500, callbacks=cb)

