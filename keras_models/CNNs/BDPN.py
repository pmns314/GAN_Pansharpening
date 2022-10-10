import os
import shutil

import keras.layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, PReLU, MaxPool2D, Lambda
from keras.optimizers import Adam
from tensorflow.python.framework.error_interpolation import interpolate

from dataset.DataGeneratorKeras import DataGenerator


def charbonnier_loss(y_true, y_pred):
    epsilon = 1e-6
    x = y_true - y_pred

    # loss = sqrt(x**2 + eps**2)
    loss = tf.math.sqrt(tf.math.square(x) + epsilon)
    # Mean over batch
    loss = tf.reduce_mean(tf.reduce_mean(loss, [1, 2, 3]))
    return loss


# Custom Keras Loss ---> Non usata perchè keras Fit non somma le singole loss e restituisce sempre 0
#
# def loss_fn(downsample):
#     def charbonnier_loss(y_true, y_pred):
#         epsilon = 1e-6
#         if downsample:
#             y_true = tf.image.resize(y_true, tf.shape(y_true)[1:3] // 2, method='nearest')
#         x = y_true - y_pred
#
#         # loss = sqrt(x**2 + eps**2)
#         loss = tf.math.sqrt(tf.math.square(x) + epsilon)
#         # Mean over batch
#         loss = tf.reduce_mean(tf.reduce_mean(loss, [1, 2, 3]))
#
#         return loss
#
#     return charbonnier_loss


class BDPN(tf.keras.Model):
    class ResBlock(keras.layers.Layer):
        def __init__(self):
            super(BDPN.ResBlock, self).__init__()
            self.conv1 = Conv2D(64, 3, padding='same', use_bias=True)
            self.conv2 = Conv2D(64, 3, padding='same', use_bias=True)
            self.prelu = PReLU(shared_axes=[1, 2])

        def call(self, inputs, **kwargs):
            x = self.conv1(inputs)
            x = self.prelu(x)
            x = self.conv2(x)
            out = inputs + x
            return out

    def __init__(self, channels, name="BDPN"):
        super(BDPN, self).__init__()
        self._model_name = name
        self.conv1 = Conv2D(64, 3, padding='same', use_bias=True)
        self.backbone1 = [BDPN.ResBlock() for _ in range(10)]
        self.backbone2 = [BDPN.ResBlock() for _ in range(10)]
        self.conv3 = Conv2D(channels, 3, padding='same', use_bias=True)
        self.conv4 = Conv2D(channels * 4, 3, padding='same', use_bias=True)
        self.conv5 = Conv2D(channels * 4, 3, padding='same', use_bias=True)

        self.maxpool = MaxPool2D(2, strides=(2, 2))  # Downsample
        self.pixelShuffle = Lambda(lambda x: tf.nn.depth_to_space(x, 2))  # Upsample

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.opt = None
        self.lambda_v = 5

    def call(self, inputs):
        try:
            pan, ms, ms_lr = inputs
        except ValueError:
            ms_lr, pan = inputs

        # Pan Feature stage 1
        pan_feature = self.conv1(pan)

        rs = pan_feature
        for layer in self.backbone1:
            rs = layer(rs)
        pan_feature1 = pan_feature + rs  # Connection absent
        pan_feature_level1 = self.conv3(pan_feature1)
        pan_feature1_out = self.maxpool(pan_feature1)

        # Pan Feature stage 2
        # Missing a Convolutional Layer
        rs = pan_feature1_out
        for layer in self.backbone2:
            rs = layer(rs)
        pan_feature2 = pan_feature1_out + rs  # Connection absent
        pan_feature_level2 = self.conv3(pan_feature2)  # Reusing Layer

        # MS Feature stage 1
        ms_feature1 = self.conv4(ms_lr)
        ms_feature_up1 = self.pixelShuffle(ms_feature1)
        ms_feature_level1 = pan_feature_level2 + ms_feature_up1

        # MS Feature Stage 2
        ms_feature2 = self.conv5(ms_feature_level1)
        ms_feature_up2 = self.pixelShuffle(ms_feature2)
        output = pan_feature_level1 + ms_feature_up2

        return output, ms_feature_level1

    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs)

    def compile(self, optimizer=None, *args, **kwargs):
        super(BDPN, self).compile()
        self.opt = optimizer

    def loss_fn(self, inputs, gt):
        sr, sr_down = self(inputs)
        gt_down = tf.image.resize(gt, tf.shape(gt)[1:3] // 2, method='nearest')
        loss1 = charbonnier_loss(sr_down, gt_down)
        loss2 = charbonnier_loss(sr, gt)

        loss = self.lambda_v * loss1 + (1.0 - self.lambda_v) * loss2
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    @property
    def name(self):
        return self._model_name

    def reset_metrics(self):
        self.loss_tracker.reset_states()

    def train_step(self, data):

        ground_truth = data[1]

        # Training
        with tf.GradientTape() as tape:
            loss_value = self.loss_fn(data[0], ground_truth)
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss_value, trainable_vars)
        self.opt.apply_gradients(zip(grads, trainable_vars))
        self.loss_tracker.update_state(loss_value)  # Add current batch loss

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        try:
            ground_truth = data[1]
        except ValueError:
            _, _, ground_truth = data[0]

        self.loss_tracker.update_state(self.loss_fn(data[0], ground_truth))

        return {
            "loss": self.loss_tracker.result()
        }


    # def train_step(self, data):
    #     original_pans, original_ms, original_ms_lr = data[0]
    #     ground_truth = data[1]
    #
    #     # Generate Data for Discriminators Training
    #     sr, sr_down = self([original_ms_lr, original_pans])
    #
    #     # Training Discriminator
    #     with tf.GradientTape() as tape:
    #         loss_value = self.loss_fn(sr, sr_down, ground_truth)
    #         trainable_vars = self.trainable_variables
    #         grads = tape.gradient(loss_value, trainable_vars)
    #     self.opt.apply_gradients(zip(grads, trainable_vars))
    #     self.loss_tracker.update_state(loss_value)  # Add current batch loss
    #
    #     return {
    #         "loss": self.loss_tracker.result()
    #     }
    #
    # def test_step(self, data):
    #     original_pans, original_ms, original_ms_lr = data[0]
    #     ground_truth = data[1]
    #
    #     # Generate Data for Discriminators Training
    #     sr, sr_down = self([original_ms_lr, original_pans])
    #
    #     loss_value = self.loss_fn(sr, sr_down, ground_truth)
    #     self.loss_tracker.update_state(loss_value)
    #
    #     return {
    #         "loss": self.loss_tracker.result()
    #     }
    #
    # def loss_fn(self, sr, sr_down, gt):
    #     gt_down = tf.image.resize(gt, tf.shape(gt) * .5, method='nearest')
    #     loss1 = charbonnier_loss(sr_down, gt_down)
    #     loss2 = charbonnier_loss(sr, gt)
    #
    #     loss = self.lambda_v * loss1 + (1.0 - self.lamda_v) * loss2
    #     return loss
    #
    # def compile(self, optimizer=None):
    #     super(BDPN, self).compile()
    #     self.opt = optimizer
    #
    # @property
    # def metrics(self):
    #     return [self.loss_tracker]
    #
    # def reset_metrics(self):
    #     self.loss_tracker.reset_states()


if __name__ == '__main__':

    train = True
    model = BDPN(8)
    path = f"..\\trained_models\\{model.name}_model"

    train_gen = DataGenerator("../../datasets/train.h5")
    val_gen = DataGenerator("../../datasets/valid.h5")
    test_gen = DataGenerator("../../datasets/test.h5", shuffle=False)

    if train is True:

        model.compile(
            optimizer=Adam(),
            loss=None
            # loss_weights=[.5, 1.0],
        )

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
    # Test Evaluate with Random data
    random_pan = np.random.randn(3, 128, 128, 1)
    random_ms = np.random.randn(3, 128, 128, 8)
    random_ms_lr = np.random.randn(3, 32, 32, 8)
    random_gt = np.random.randn(3, 128, 128, 8)
    model.evaluate([random_ms_lr, random_pan], random_gt)

    # Test Predict
    pan, ms, ms_lr = next(test_gen.generate_batch())[0]
    gen = model.predict([ms_lr, pan])

    # Test Evaluate with generator
    model.evaluate(test_gen)
