import tempfile
from pathlib import Path

import dvc.api  # type: ignore
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report  # type: ignore
from sklearn.metrics import confusion_matrix


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
    model = tf.keras.models.load_model(model_file)
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
