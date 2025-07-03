import keras  # type: ignore
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers, models

from .config import Config


@keras.saving.register_keras_serializable()
def SSIMLoss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def build_model(
        num_labels: int,
) -> models.Model:
    model = models.Sequential([
        layers.Input(shape=(None, None, 1)),
        layers.Resizing(128, 64),
        layers.Normalization(),
        layers.Conv2D(8, 3, activation='relu', padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation='relu', padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(16, 3, activation='relu', padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(16, 3, activation='relu', padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(16, 3, activation='relu', padding="same"),
        layers.MaxPooling2D((1, 4)),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_labels),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model
