import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers, models

from .config import Config


def SSIMLoss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def build_model(
        input_shape: tuple[int, int, int],
        num_labels: int,
        class_loss_weight: float = 0.5,
        recon_loss_weight: float = 0.5,
) -> models.Model:
    # Нормализация входных данных
    norm_layer = layers.Normalization()

    inputs = layers.Input(shape=input_shape)
    x = norm_layer(inputs)

    # Энкодер: последовательность свёрточных слоёв и пулинга
    x = layers.Conv2D(8, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D((1, 4))(x)
    x = layers.Flatten()(x)
    # Переход к полносвязной части

    encoded = layers.Dropout(0.25)(x)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dropout(0.25)(encoded)

    # Ветвь классификации
    class_output = layers.Dense(num_labels, name='classification_output')(
        encoded)

    # Ветвь декодера (восстановление спектрограммы)
    x2 = layers.Concatenate(axis=1)([x, class_output])
    x2 = layers.Dropout(0.25)(x2)
    x2 = layers.Dense(128, activation='relu')(x2)

    x2 = layers.Reshape((8, 1, 16))(x2)
    x2 = layers.Conv2DTranspose(16, kernel_size=2, strides=(1, 4),
                                activation='relu', padding='same')(x2)
    x2 = layers.Conv2DTranspose(16, kernel_size=2, strides=2,
                                activation='relu', padding='same')(x2)
    x2 = layers.Conv2DTranspose(16, kernel_size=2, strides=2,
                                activation='relu', padding='same')(x2)
    x2 = layers.Conv2DTranspose(16, kernel_size=2, strides=2,
                                activation='relu', padding='same')(x2)
    decoder_output = layers.Conv2DTranspose(input_shape[-1], kernel_size=2,
                                            strides=2,
                                            activation='relu', padding='same',
                                            name='reconstruction_output')(x2)

    # Модель с двумя выходами: для классификации
    # и для восстановления спектрограммы
    model: models.Model = models.Model(inputs=inputs,
                                       outputs=[class_output, decoder_output])

    # Задаём функции потерь для каждой из ветвей.
    # Для классификации используем SparseCategoricalCrossentropy
    #     (с from_logits=True, если выход не проходит softmax)
    # Для восстановления – MeanAbsolutePercentageError
    #     (можно использовать и другую функцию, например, MSE)
    loss_dict = {
        'classification_output':
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # 'reconstruction_output': tf.keras.losses.MeanSquaredError(),
        'reconstruction_output': SSIMLoss,
    }

    # Взвешиваем вклад каждой ошибки
    loss_weights = {
        'classification_output': class_loss_weight,
        'reconstruction_output': recon_loss_weight
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        # optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss=loss_dict,  # type: ignore
        loss_weights=loss_weights,
        metrics={'classification_output': 'accuracy'})

    return model
