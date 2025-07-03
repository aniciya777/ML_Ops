import os
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import tensorflow as tf

from train.config import Config  # type: ignore
from validation.utils import fetch_model_at_rev, evaluate_model


class OneClassResult(NamedTuple):
    precision: float
    recall: float
    f1: float


class RevisionResult:
    def __init__(self, num_folds: int, classes: Sequence[str]) -> None:
        self._num_folds = num_folds
        self._classes = classes
        self._acc = 0.0
        self._precisions = {cls: 0.0 for cls in classes}
        self._recall = {cls: 0.0 for cls in classes}
        self._f1 = {cls: 0.0 for cls in classes}

    def add(self, report: dict) -> 'RevisionResult':
        self._acc += report["accuracy"]
        for i, cls in enumerate(self._classes):
            s_i = str(i)
            self._precisions[cls] += report[s_i]["precision"]
            self._recall[cls] += report[s_i]["recall"]
            self._f1[cls] += report[s_i]["f1-score"]
        return self

    @property
    def accuracy(self) -> float:
        return self._acc / self._num_folds

    def __getitem__(self, item: str) -> OneClassResult:
        return OneClassResult(
            self._precisions[item] / self._num_folds,
            self._recall[item] / self._num_folds,
            self._f1[item] / self._num_folds,
        )


def main() -> None:
    SPEC_DS_DIR = Path("data/spec_ds")
    TEST_DS_PATH = SPEC_DS_DIR / "test_specs"
    LABELS_PATH = SPEC_DS_DIR / "label_names.npy"

    MODEL_DIR = Path("data/models")

    with open("data/сomparison_of_revisions.txt") as f:
        REVISIONS = map(str.strip, f.readlines())

    dummy = tf.data.Dataset.load(str(TEST_DS_PATH), compression="GZIP")
    spec = dummy.element_spec
    ds = tf.data.Dataset.load(str(TEST_DS_PATH),
                              element_spec=spec,
                              compression="GZIP")
    test_ds = ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    label_names = np.load(LABELS_PATH)
    model_files = [
        MODEL_DIR / f"model{i + 1}.keras"
        for i in range(Config.NUM_FOLDS)
    ]

    results: dict[str, RevisionResult] = {}
    for rev in REVISIONS:
        print(f"\n Process the revision `{rev}`")
        results[rev] = RevisionResult(Config.NUM_FOLDS, label_names)
        for filename in model_files:
            local_model = fetch_model_at_rev(filename, rev)
            report, cm = evaluate_model(local_model, test_ds)
            results[rev].add(report)
            os.remove(local_model)

    with open("comparison_versions.md", "w", encoding='utf-8') as mf:
        mf.write("# Сравнение версий модели на тесте\n\n")
        for rev, res in results.items():
            mf.write(f"## Revision `{rev}`\n\n")
            mf.write(f"**Accuracy**: {res.accuracy:.4f}  \n\n")
            mf.write("**P / R / F1 по классам:**  \n")
            for cls in label_names:
                mf.write(f"- `{cls}`: P={res[cls].precision:.2f}, "
                         f"R={res[cls].recall:.2f}, F1={res[cls].f1:.2f}\n")
    #         mf.write("\nМатрица ошибок:\n\n```\n")
    #         mf.write(np.array2string(res["cm"]))
    #         mf.write("\n```\n\n---\n\n")


if __name__ == '__main__':
    main()
