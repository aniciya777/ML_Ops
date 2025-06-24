import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from train.config import Config  # type: ignore

from .utils import evaluate_model, fetch_model_at_rev


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
    results = {}
    for rev in REVISIONS:
        print(f"\n Process the revision `{rev}`")
        local_model = fetch_model_at_rev(MODEL_DIR / "model1.keras", rev)
        print(f"  downloaded in {local_model}")

        report, cm = evaluate_model(local_model, test_ds)
        results[rev] = {"report": report, "cm": cm}
        os.remove(local_model)

    # 5) пишем Markdown-отчет
    with open("comparison_versions.md", "w", encoding='utf-8') as mf:
        mf.write("# Сравнение версий модели на тесте\n\n")
        for rev, res in results.items():
            mf.write(f"## Revision `{rev}`\n\n")
            acc = res["report"]["accuracy"]
            mf.write(f"**Accuracy**: {acc:.4f}  \n\n")
            mf.write("**P / R / F1 по классам:**  \n")
            for i, cls in enumerate(label_names):
                m = res["report"][str(i)]
                mf.write(f"- `{cls}`: P={m['precision']:.2f}, "
                         f"R={m['recall']:.2f}, F1={m['f1-score']:.2f}\n")
            mf.write("\nМатрица ошибок:\n\n```\n")
            mf.write(np.array2string(res["cm"]))
            mf.write("\n```\n\n---\n\n")


if __name__ == '__main__':
    main()
