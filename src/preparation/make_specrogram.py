import logging
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore
from clearml import Task, TaskTypes  # type: ignore

from train.config import Config  # type: ignore
from visualisation.utils import (  # type: ignore
    plot_sample_spectrogram,
    plot_samples_waveforms,
)

from .utils import make_spec_ds, squeeze

logging.basicConfig(level=logging.INFO)


def main() -> None:
    data_dir = "data/output/ML"
    batch_size: int | None = None
    validation_split = 0.1

    task = Task.init(
        project_name=Config.PROJECT_NAME,
        task_name='make spectrogram',
        task_type=TaskTypes.data_processing,
        auto_connect_frameworks={
            'matplotlib': True,
            'detect_repository': False,
        }
    )
    logger = task.get_logger()
    task.set_progress(0)
    logger.report_text(f'directory: {data_dir}')
    if batch_size is None:
        logger.report_text('batch_size is None')
    else:
        logger.report_single_value('batch_size', batch_size)
    logger.report_single_value('validation_split', validation_split)
    logger.report_single_value('seed', Config.SEED)
    logger.report_single_value('output_sequence_length', Config.AUDIO_LENGTH)

    raw_train_ds, raw_test_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=Config.SEED,
        output_sequence_length=Config.AUDIO_LENGTH,
        subset='both'
    )
    task.set_progress(50)

    # Извлекаем имена классов до любого .map()
    label_names = np.array(raw_train_ds.class_names, dtype='<U50')
    out_dir = Path("data/spec_ds")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Loaded classes: {label_names}")
    np.save(out_dir / "label_names.npy", label_names)
    task.upload_artifact("label_names", out_dir / "label_names.npy")

    # 2) Применяем squeeze и строим спектрограммы
    train_ds = raw_train_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = raw_test_ds.map(squeeze, tf.data.AUTOTUNE)
    plot_samples_waveforms(train_ds, label_names, 9)

    plot_sample_spectrogram(train_ds, label_names, 0)
    train_spec_ds = make_spec_ds(train_ds)  # (spec, label)
    test_spec_ds = make_spec_ds(test_ds)

    # 3) Сохраняем без батчей
    train_path = out_dir / "train_specs"
    test_path = out_dir / "test_specs"

    train_spec_ds.save(str(train_path), compression="GZIP")
    test_spec_ds.save(str(test_path), compression="GZIP")
    task.upload_artifact("train_specs", train_path)
    task.upload_artifact("test_specs", test_path)
    task.set_progress(100)
    task.close()


if __name__ == '__main__':
    main()
