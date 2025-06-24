import tempfile
from pathlib import Path

import dvc.api  # type: ignore
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report  # type: ignore
from sklearn.metrics import confusion_matrix

from train.model import SSIMLoss  # type: ignore


def fetch_model_at_rev(dvc_path: Path, rev: str) -> str:
    """
    Скачивает файл models/model.keras из DVC-кэша
    для указанной ревизии rev и возвращает локальный путь.
    """
    with dvc.api.open(
            path=str(dvc_path),
            repo=".",
            remote="myremote",
            rev=rev,
            mode='rb'
    ) as src:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        tmp.write(src.read())
        tmp.close()
        return tmp.name


def evaluate_model(model_file: str, test_ds):
    model = tf.keras.models.load_model(
        model_file,
        custom_objects={"SSIMLoss": SSIMLoss}
    )
    y_true, y_pred = [], []
    for x_batch, label_batch in test_ds:

        # model.predict вернёт [class_preds, recon_preds]
        outputs = model.predict(x_batch, verbose=0)
        if isinstance(outputs, (list, tuple)):
            class_preds = outputs[0]
        else:
            class_preds = outputs

        # собираем «правильные» и «предсказанные» метки
        y_true.extend(label_batch.numpy().tolist())
        y_pred.extend(np.argmax(class_preds, axis=1).tolist())

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
