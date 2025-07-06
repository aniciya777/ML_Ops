import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # type: ignore
from numpy._typing import NDArray

from preparation.utils import get_spectrogram  # type: ignore
from train.config import Config  # type: ignore


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


def plot_samples_waveforms(
    ds: tf.data.Dataset[tuple[tf.Tensor, tf.Tensor]],
    label_names: NDArray[np.str_],
    count: int,
    columns: int = 3
) -> None:
    rows = (count + columns - 1) // columns
    plt.figure(figsize=(16, 3 * rows))
    for idx, (audio, label) in enumerate(ds.take(count).as_numpy_iterator()):
        plt.subplot(rows, columns, idx + 1)
        plt.plot(audio)
        plt.title(label_names[label])
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.ylim([-1.1, 1.1])
    plt.tight_layout()
    plt.title(f"{count} samples of waveforms")
    plt.show()


def plot_sample_spectrogram(
    ds: tf.data.Dataset[tuple[tf.Tensor, tf.Tensor]],
    label_names: NDArray[np.str_],
    sample_index: int
) -> None:
    waveform, label_index = next(iter(ds.skip(sample_index).take(1)))
    spectrogram = get_spectrogram(waveform)
    label = label_names[label_index]

    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(int(waveform.shape[0]))  # type: ignore[arg-type]
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, Config.AUDIO_LENGTH])

    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.suptitle(label.title())
    plt.title(f"{sample_index} sample of spectrogram")
    plt.show()
