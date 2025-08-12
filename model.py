import settings
import os
import time
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras


def dice_coefficient(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sørensen Dice
    where T is ground truth mask and P is the prediction mask
    """
    prediction = keras.backend.round(prediction)  # Round to 0 or 1
    return soft_dice_coefficient(target, prediction, axis, smooth)


def soft_dice_coefficient(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Soft Sørensen Dice - Don't round the predictions
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)

    numerator = tf.constant(2.0) * intersection + smooth
    denominator = union + smooth

    return tf.reduce_mean(numerator / denominator)


def dice_coefficient_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sørensen (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)

    return -tf.math.log(2.0 * numerator) + tf.math.log(denominator)


def combined_dice_ce_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    dice_loss = dice_coefficient_loss(target, prediction, axis, smooth)
    bce_loss = keras.losses.binary_crossentropy(target, prediction)
    return (
            settings.WEIGHT_DICE_LOSS * dice_loss
            + (1 - settings.WEIGHT_DICE_LOSS) * bce_loss
    )


def sensitivity(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sensitivity
    """
    true_positives = tf.reduce_sum(target * prediction, axis=axis)
    possible_positives = tf.reduce_sum(target, axis=axis)

    return tf.reduce_mean((true_positives + smooth) / (possible_positives + smooth))


def specificity(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Specificity
    """
    true_negatives = tf.reduce_sum((1 - target) * (1 - prediction), axis=axis)
    false_positives = tf.reduce_sum((1 - target) * prediction, axis=axis)
    return tf.reduce_mean(true_negatives / (true_negatives + false_positives + smooth))


class UNet2D(object):
    def __init__(
            self,
            channels_first=settings.CHANNELS_FIRST,
            fms=settings.FEATURE_MAPS,
            output_path=settings.OUTPUT_PATH,
            inference_filename=settings.INFERENCE_FILENAME,
            block_time=settings.BLOCK_TIME,
            num_threads=settings.NUM_INTRA_THREADS,
            learning_rate=settings.LEARNING_RATE,
            num_inter_threads=settings.NUM_INTER_THREADS,
            use_up_sampling=settings.USE_UP_SAMPLING,
            use_dropout=settings.USE_DROPOUT,
            dropout_rate=settings.DROPOUT_RATE,
            print_model=settings.PRINT_MODEL,
    ):
        self.channels_first = channels_first
        self.fms = fms
        self.output_path = output_path
        self.inference_filename = inference_filename
        self.block_time = block_time
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.num_inter_threads = num_inter_threads
        self.use_up_sampling = use_up_sampling
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.print_model = print_model

        self.num_input_channels = None
        self.input_shape = None
        self.model = None

        if self.channels_first:
            """
            Use NCHW format for data
            """
            self.concat_axis = 1
            self.data_format = "channels_first"
        else:
            """
            Use NHWC format for data
            """
            self.concat_axis = -1
            self.data_format = "channels_last"

        print("Data format = " + self.data_format)
        keras.backend.set_image_data_format(self.data_format)

        self.metrics = [
            dice_coefficient,
            soft_dice_coefficient,
            sensitivity,
            specificity,
        ]
        self.loss = dice_coefficient_loss
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.learning_rate,
        )
        self.custom_objects = {
            "combined_dice_ce_loss": combined_dice_ce_loss,
            "dice_coefficient_loss": dice_coefficient_loss,
            "dice_coefficient": dice_coefficient,
            "soft_dice_coefficient": soft_dice_coefficient,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def unet_model(
            self, images_shape, masks_shape, dropout=settings.DROPOUT_RATE, final=False
    ):
        """
        =====================================================================
         UNet Model
         Based on https://arxiv.org/abs/1505.04597

         The default uses UpSampling2D (nearest neighbors interpolation) in
         the decoder path. The alternative is to use Transposed Convolution.
        =====================================================================
        """

        def conv_block(i, filters, name_prefix, dropout=None):
            x = keras.layers.Conv2D(name=f"{name_prefix}a", filters=filters, **params)(
                i
            )
            if dropout:
                x = keras.layers.SpatialDropout2D(dropout)(x)
            x = keras.layers.Conv2D(name=f"{name_prefix}b", filters=filters, **params)(
                x
            )
            return x

        def up_sampling_block(i, skip_connection, filters, name_prefix):
            if self.use_up_sampling:
                up = keras.layers.UpSampling2D(name=f"up{name_prefix}", size=(2, 2))(i)
            else:
                up = keras.layers.Conv2DTranspose(
                    name=f"transconv{name_prefix}", filters=filters, **params_trans
                )(i)
            return keras.layers.concatenate(
                [up, skip_connection],
                axis=self.concat_axis,
                name=f"concat{name_prefix}",
            )

        if not final:
            print(
                "Using UpSampling2D"
                if self.use_up_sampling
                else "Using Transposed Convolution"
            )

        num_chan_in = images_shape[self.concat_axis]
        num_chan_out = masks_shape[self.concat_axis]
        self.input_shape = images_shape
        self.num_input_channels = num_chan_in

        inputs = keras.layers.Input(self.input_shape, name="MRImages")

        # Convolution parameters
        params = dict(
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_uniform",
        )
        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2), strides=(2, 2), padding="same")

        encodeA = conv_block(inputs, self.fms, "encodeA")
        poolA = keras.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

        encodeB = conv_block(poolA, self.fms * 2, "encodeB")
        poolB = keras.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

        encodeC = conv_block(
            poolB, self.fms * 4, "encodeC", dropout if self.use_dropout else None
        )
        poolC = keras.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

        encodeD = conv_block(
            poolC, self.fms * 8, "encodeD", dropout if self.use_dropout else None
        )
        poolD = keras.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

        encodeE = conv_block(poolD, self.fms * 16, "encodeE")

        concatD = up_sampling_block(encodeE, encodeD, self.fms * 8, "D")
        decodeC = conv_block(concatD, self.fms * 8, "decodeC")

        concatC = up_sampling_block(decodeC, encodeC, self.fms * 4, "C")
        decodeB = conv_block(concatC, self.fms * 4, "decodeB")

        concatB = up_sampling_block(decodeB, encodeB, self.fms * 2, "B")
        decodeA = conv_block(concatB, self.fms * 2, "decodeA")

        concatA = up_sampling_block(decodeA, encodeA, self.fms, "A")
        convOut = conv_block(concatA, self.fms, "convOut")

        prediction = keras.layers.Conv2D(
            name="PredictionMask",
            filters=num_chan_out,
            kernel_size=(1, 1),
            activation="sigmoid",
        )(convOut)

        model = keras.models.Model(inputs=[inputs], outputs=[prediction], name="UNet2D")

        if final:
            model.trainable = False
        else:
            model.compile(
                optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
            )
            if self.print_model:
                model.summary()

        return model

    def get_callbacks(self):
        model_filename = os.path.join(self.output_path, self.inference_filename)

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            model_filename, verbose=0, monitor="val_loss", save_best_only=True
        )

        directoryName = "unet_block{}_inter{}_intra{}".format(
            self.block_time, self.num_threads, self.num_inter_threads
        )

        # Tensorboard callbacks
        if self.use_up_sampling:
            tensorboard_filename = os.path.join(
                self.output_path,
                "keras_tensorboard_upsampling/{}".format(directoryName),
            )
        else:
            tensorboard_filename = os.path.join(
                self.output_path,
                "keras_tensorboard_transposed/{}".format(directoryName),
            )

        tensorboard_checkpoint = keras.callbacks.TensorBoard(
            log_dir=tensorboard_filename, write_graph=True, write_images=True
        )

        early_stopping = keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        )

        csv_logger = keras.callbacks.CSVLogger("training_log.csv", append=True)

        return model_filename, [
            model_checkpoint,
            early_stopping,
            reduce_lr,
            tensorboard_checkpoint,
            csv_logger,
        ]

    def evaluate_model(self, model_filename, ds_test):
        model = keras.models.load_model(
            model_filename, custom_objects=self.custom_objects
        )
        print("Evaluating model on test dataset. Please wait...")
        metrics = model.evaluate(ds_test, verbose=1)
        for idx, metric in enumerate(metrics):
            print("Test dataset {} = {:.4f}".format(model.metrics_names[idx], metric))

    def create_model(self, images_shape, masks_shape, dropout=settings.DROPOUT_RATE, final=False):
        self.model = self.unet_model(images_shape, masks_shape, dropout=dropout, final=final)
        return self.model

    def load_model(self, model_filename):
        self.model = keras.models.load_model(model_filename, custom_objects=self.custom_objects)
        return self.model
