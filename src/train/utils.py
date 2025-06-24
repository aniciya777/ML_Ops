import logging

import tensorflow as tf

from .config import Config


def squeeze(
        audio: tf.Tensor,
        labels: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    spectrogram = tf.image.resize(spectrogram, [128, 64])
    nonzero_elements = tf.boolean_mask(spectrogram, spectrogram != 0)
    nonzero_median = tf.sort(nonzero_elements)[
        tf.shape(nonzero_elements)[0] // 4]
    spectrogram = tf.where(spectrogram == 0, nonzero_median, spectrogram)
    return spectrogram


def make_spec_ds(ds: tf.Tensor) -> tf.Tensor:
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)


def scheduler(epoch: int, lr: float) -> float:
    if epoch == 0:
        return Config.LEARNING_RATE
    if epoch > 0 and epoch % 500 == 0:
        lr *= 0.5
        logging.info(f'Change lr to {lr}')
    return lr
