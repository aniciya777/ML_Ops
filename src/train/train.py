import logging
import os
from pathlib import Path

import neptune
import numpy as np
import tensorflow as tf
from neptune import Run
from neptune.integrations import tensorflow_keras as n_tf  # type: ignore
from sklearn.model_selection import KFold  # type: ignore[import-untyped]

from .config import Config
from .model import build_model
from .utils import scheduler

logging.basicConfig(level=logging.INFO)


def train(run: Run) -> None:
    logging.info(tf.config.list_physical_devices('GPU'))
    out_dir = Path("data/spec_ds")
    train_path = out_dir / "train_specs"

    loaded_ds = tf.data.Dataset.load(
        str(train_path),
        compression="GZIP"
    )
    label_names = np.load(out_dir / "label_names.npy")
    logging.info(f"Loaded classes: {label_names}")

    input_shape = (Config.SPECTROGRAM_WIDTH, Config.SPECTROGRAM_HEIGHT, 1)
    logging.info(f"input_shape: {input_shape}")
    num_labels = len(label_names)

    all_data = list(loaded_ds)  # оставляем как список
    kf = KFold(
        n_splits=Config.NUM_FOLDS,
        shuffle=True,
        random_state=Config.SEED
    )
    class_loss_weight = 1
    recon_loss_weight = 20

    # Логируем параметры модели в Neptune

    run['training/model/params'] = {
        'learning_rate': Config.LEARNING_RATE,
        'epoch': Config.EPOCHS,
        'batch': Config.BATCH_SIZE,
        'num_folds': Config.NUM_FOLDS,
        'input_shape': str(input_shape),
        'class_loss_weight': class_loss_weight,
        'recon_loss_weight': recon_loss_weight,
    }

    acc_per_fold = []
    loss_per_fold = []
    my_models = []
    histories = []

    for fold_no, (train_index, val_index) in enumerate(kf.split(all_data), 1):
        print(f"\n--- Fold {fold_no} ---")

        # Используем list comprehension для выборки данных
        train_data = [all_data[i] for i in train_index]
        val_data = [all_data[i] for i in val_index]

        # Разъединяем спектрограммы и метки
        train_specs, train_labels = zip(*train_data)
        val_specs, val_labels = zip(*val_data)

        # Создаем tf.data.Dataset из списков
        train_ds_cv = tf.data.Dataset.from_tensor_slices(
            (list(train_specs), (list(train_labels))))
        val_ds_cv = tf.data.Dataset.from_tensor_slices(
            (list(val_specs), (list(val_labels))))

        # Батчим, кешируем и префетчим
        train_ds_cv = (
            train_ds_cv
            .batch(Config.BATCH_SIZE)
            .cache()
            .shuffle(1_000)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds_cv = (
            val_ds_cv
            .batch(Config.BATCH_SIZE)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        model = build_model(num_labels)
        my_models.append(model)

        # Создаем Neptune callback для отслеживания метрик в текущем фолде
        neptune_cbk = n_tf.NeptuneCallback(  # type: ignore
            run=run,
            base_namespace=f"training/model/folds/{fold_no}/metrics"
        )
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler,
                                                               verbose=0)

        history = model.fit(
            train_ds_cv,
            validation_data=val_ds_cv,
            epochs=Config.EPOCHS,
            callbacks=[lr_callback, neptune_cbk],
            verbose=0
        )

        scores: list[float] | float = model.evaluate(val_ds_cv, verbose=0)
        if isinstance(scores, float):
            scores = [scores]
        print(scores)
        print(
            f"Fold {fold_no} - Loss: {scores[0]:.4f} "
            f"- Accuracy: {scores[1] * 100:.2f}%")

        loss_per_fold.append(scores[0])
        acc_per_fold.append(scores[1])
        histories.append(history)

        # Логируем итоговые метрики фолда в Neptune
        run[f"training/model/folds/{fold_no}/final_loss"] = scores[0]
        run[f"training/model/folds/{fold_no}/final_accuracy"] = scores[1]

    for i in range(Config.NUM_FOLDS):
        my_models[i].save(os.path.join(
            'data', 'models',
            f'model{i + 1}.keras'
        ))
    avg_accuracy = np.mean(acc_per_fold)
    avg_loss = np.mean(loss_per_fold)
    run["training/avg_accuracy"] = avg_accuracy
    run["training/avg_loss"] = avg_loss


def before_run() -> Run:
    tf.random.set_seed(Config.SEED)
    np.random.seed(Config.SEED)
    return neptune.init_run(
        project=Config.NEPTUNE,
        api_token=Config.NEPTUNE_TOKEN
    )


def main() -> None:
    train(before_run())


if __name__ == '__main__':
    main()
