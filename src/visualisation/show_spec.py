from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def main() -> None:
    out_dir = Path("data/spec_ds")
    label_names = np.load(out_dir / "label_names.npy")
    train_path = out_dir / "train_specs"
    spectrogram_ds = tf.data.Dataset.load(
        str(train_path),
        compression="GZIP"
    )

    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(spectrogram.numpy(), ax)
        ax.set_title(label_names[label_id.numpy()])
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
