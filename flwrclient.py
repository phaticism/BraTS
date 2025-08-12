import flwr as fl
import os
import datetime
import settings
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from model import UNet2D
from dataloader import DatasetGenerator, get_decathlon_filelist


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, unet_model, train_data, val_data, test_data, settings):
        if not hasattr(unet_model, "model") or unet_model.model is None:
            raise ValueError("The UNet2D model must be created before initializing FlowerClient.")
        self.unet_model = unet_model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.settings = settings

    def get_parameters(self, config=None):
        return self.unet_model.model.get_weights()

    def set_parameters(self, parameters):
        self.unet_model.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Configure callbacks
        model_filename = os.path.join(
            self.settings.OUTPUT_PATH, f"node_{self.settings.NODE_ID}", self.settings.INFERENCE_FILENAME
        )
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                model_filename,
                monitor="val_loss",
                save_best_only=True,
                verbose=0
            ),
            # EarlyStopping(
            #     monitor="val_loss",
            #     patience=4,
            #     restore_best_weights=True
            # ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]

        # Train the model
        history = self.unet_model.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=int(config.get("epochs", self.settings.EPOCHS)),
            batch_size=int(config.get("batch_size", self.settings.BATCH_SIZE)),
            callbacks=callbacks,
            verbose=1
        )

        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, dice_coef, soft_dice, sensitivity, specificity = self.unet_model.model.evaluate(
            self.test_data,
            verbose=1
        )
        return loss, len(self.test_data), {
            "dice_coefficient": dice_coef,
            "soft_dice": soft_dice,
            "sensitivity": sensitivity,
            "specificity": specificity
        }


def client_fn(context) -> fl.client.Client:
    """Create a Flower client representing a single node."""
    
    # Load data partition for this client
    partition_id = context.node_config["partition-id"]
    # Format partition_id to 3 digits (e.g., 0 -> "001", 1 -> "002", 2 -> "003")
    partition_id = f"{partition_id + 1:03d}"
    partition_path = os.path.join(settings.TRAIN_DATA_PATH, partition_id)
    
    train_files, validate_files, test_files = get_decathlon_filelist(
        data_path=partition_path,
        seed=settings.SEED,
        split=settings.TRAIN_TEST_SPLIT,
    )

    # Create data generators
    train_data = DatasetGenerator(
        train_files,
        batch_size=settings.BATCH_SIZE,
        crop_dim=(settings.CROP_DIM, settings.CROP_DIM),
        augment=settings.USE_AUGMENTATION,
        seed=settings.SEED,
        dim=3
    )
    
    val_data = DatasetGenerator(
        validate_files,
        batch_size=settings.BATCH_SIZE,
        crop_dim=(settings.CROP_DIM, settings.CROP_DIM),
        augment=False,
        seed=settings.SEED,
        dim=3
    )
    
    test_data = DatasetGenerator(
        test_files,
        batch_size=settings.BATCH_SIZE,
        crop_dim=(settings.CROP_DIM, settings.CROP_DIM),
        augment=False,
        seed=settings.SEED,
        dim=3
    )

    # Initialize the model
    unet_model = UNet2D(
        fms=settings.FEATURE_MAPS,
        learning_rate=settings.LEARNING_RATE,
        use_dropout=settings.USE_DROPOUT,
        use_up_sampling=settings.USE_UP_SAMPLING,
        dropout_rate=settings.DROPOUT_RATE
    )
    
    # Create the model with proper input/output shapes
    unet_model.create_model(
        train_data.get_input_shape(),
        train_data.get_output_shape()
    )

    # Assign node-specific settings
    client_settings = settings
    client_settings.NODE_ID = partition_id

    # Create and return the Flower client
    return FlowerClient(unet_model, train_data, val_data, test_data, client_settings)
