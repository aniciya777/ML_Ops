import argparse
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore
from sklearn.model_selection import KFold  # type: ignore

from clearml import Task, TaskTypes  # type: ignore
from train.callbaks import ClearMLLogger  # type: ignore

from .config import Config
from .model import build_model
from .utils import scheduler

logging.basicConfig(level=logging.INFO)


def train() -> None:
    task = Task.init(
        project_name=Config.PROJECT_NAME,
        task_name='train model',
        auto_connect_frameworks={
            'tensorflow': True,
            'tensorboard': True,
            'detect_repository': False,
        },
        task_type=TaskTypes.training
    )
    task.set_progress(0)
    logger = task.get_logger()
    logging.info(tf.config.list_physical_devices('GPU'))
    out_dir = Path("data/spec_ds")
    train_path = out_dir / "train_specs"

    loaded_ds = tf.data.Dataset.load(
        str(train_path),
        compression="GZIP"
    )

    label_names = np.load(out_dir / "label_names.npy")
    logging.info(f"Loaded classes: {label_names}")
    task.upload_artifact("label_names", label_names)

    input_shape = (Config.SPECTROGRAM_WIDTH, Config.SPECTROGRAM_HEIGHT, 1)
    logging.info(f"input_shape: {input_shape}")
    num_labels = len(label_names)

    all_data = list(loaded_ds)  # оставляем как список
    kf = KFold(
        n_splits=Config.NUM_FOLDS,
        shuffle=True,
        random_state=Config.SEED
    )

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
            .shuffle(
                buffer_size=train_ds_cv.cardinality(),
                seed=Config.SEED
            )
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

        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        history = model.fit(
            train_ds_cv,
            validation_data=val_ds_cv,
            epochs=Config.EPOCHS,
            callbacks=[
                lr_callback,
                ClearMLLogger(
                    task=task,
                    fold_index=fold_no - 1,
                    folds_count=Config.NUM_FOLDS,
                    count_epochs=Config.EPOCHS
                )
            ],
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

    for i in range(Config.NUM_FOLDS):
        model_path = os.path.join(
            Config.output_dir,  # type: ignore
            f'model{i + 1}.keras'
        )
        my_models[i].save(model_path)
        task.upload_artifact(f'model{i + 1}', model_path)

    avg_accuracy = sum(acc_per_fold) / Config.NUM_FOLDS
    avg_loss = sum(loss_per_fold) / Config.NUM_FOLDS
    logger.report_single_value("avg_accuracy", avg_accuracy)
    logger.report_single_value("avg_loss", avg_loss)
    task.set_progress(100)
    task.close()


def before_run():
    tf.random.set_seed(Config.SEED)
    np.random.seed(Config.SEED)
    parser = argparse.ArgumentParser(
        description="Train a models with cross-validation"
    )
    parser.add_argument('--epochs', '-e', type=int,
                        default=Config.EPOCHS)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=Config.BATCH_SIZE)
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=Config.LEARNING_RATE)
    parser.add_argument('--folds', '-f', type=int,
                        default=Config.NUM_FOLDS)
    parser.add_argument('--output-dir', '-o', type=str,
                        default='data/models')
    args = parser.parse_args()
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    Config.NUM_FOLDS = args.folds
    Config.output_dir = args.output_dir


def main() -> None:
    before_run()
    train()


if __name__ == '__main__':
    main()
