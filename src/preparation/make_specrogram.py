from pathlib import Path

import numpy as np
import tensorflow as tf

from train.config import Config  # type: ignore

from .utils import make_spec_ds, squeeze


def main() -> None:
    data_dir = "data/output"

    raw_train_ds, raw_test_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=None,
        validation_split=0.1,
        seed=Config.SEED,
        output_sequence_length=Config.AUDIO_LENGTH,
        subset='both'
    )

    # Извлекаем имена классов до любого .map()
    label_names = np.array(raw_train_ds.class_names, dtype='<U50')
    out_dir = Path("data/spec_ds")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "label_names.npy", label_names)

    # 2) Применяем squeeze и строим спектрограммы
    train_ds = raw_train_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = raw_test_ds.map(squeeze, tf.data.AUTOTUNE)

    train_spec_ds = make_spec_ds(train_ds)  # (spec, label)
    test_spec_ds = make_spec_ds(test_ds)

    # 3) Сохраняем без батчей
    train_path = out_dir / "train_specs"
    test_path = out_dir / "test_specs"

    train_spec_ds.save(str(train_path), compression="GZIP")
    test_spec_ds.save(str(test_path), compression="GZIP")


if __name__ == '__main__':
    main()
