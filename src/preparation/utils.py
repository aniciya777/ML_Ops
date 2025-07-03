import os
import shutil
import subprocess
import sys
import wave
from os import makedirs, path, walk
from pathlib import Path

import librosa
import numpy as np
import soundfile  # type: ignore
import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.io import wavfile


# Функция для конвертации ogg в wav
def convert_ogg_to_wav(
        input_file: Path,
        output_file: Path
) -> None:
    try:
        audio = AudioSegment.from_file(input_file, format="ogg")
        audio.export(output_file, format="wav")
    except Exception as e:
        print(f"convert_ogg_to_wav: Ошибка при конвертации: {e}")


def convert_mp3_to_wav(
        input_file: Path,
        output_file: Path
) -> None:
    try:
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_file, format="wav")
    except Exception as e:
        print(f"convert_mp3_to_wav: Ошибка при конвертации: {e}")


def convert_opus_to_wav(
        input_file: Path,
        output_file: Path
) -> None:
    try:
        if not path.exists(input_file):
            print(f"convert_opus_to_wav: Файл {input_file} не найден!")
            return
        command: list[str | Path] = [
            "ffmpeg",
            "-i", input_file,
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            output_file
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"convert_opus_to_wav: Ошибка FFmpeg: {e}")
    except Exception as e:
        print(f"convert_opus_to_wav: Другая ошибка: {e}")


def move_file(
        source: Path,
        destination: Path
) -> None:
    try:
        shutil.copy2(source, destination)
    except Exception as e:
        print(f"move_file: Ошибка при перемещении файла: {e}")


def convert_wav_to_16bit(
        input_file: Path,
        output_file: Path
) -> None:
    try:
        # Загружаем WAV файл
        audio = AudioSegment.from_wav(input_file)
        # Преобразуем в 16-битный формат
        audio = audio.set_sample_width(2)  # 2 байта = 16 бит
        # Сохраняем как 16-битный WAV
        audio.export(output_file, format="wav")
    except Exception as e:
        print("convert_wav_to_16bit: Ошибка при конвертации файла "
              f"{input_file}: {e}")


def convert_stereo_to_mono(
        input_file: Path,
        output_file: Path
) -> None:
    try:
        with wave.open(str(input_file), "rb") as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(params.nframes)

        # Если аудио стерео, усредняем каналы
        if params.nchannels == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_data = audio_data.mean(axis=1).astype(np.int16)
        else:
            mono_data = np.frombuffer(frames, dtype=np.int16)

        # Создаём выходную директорию, если она не существует
        makedirs(path.dirname(output_file), exist_ok=True)

        wavfile.write(output_file, params.framerate, mono_data)
    except Exception as e:
        print("convert_stereo_to_mono: Ошибка при конвертации файла "
              f"{input_file}: {e}")


def remove_silence(
        input_path: Path,
        output_path: Path,
        silence_thresh: int = -50,
        min_silence_len: int = 100
) -> None:
    """Удаляет тишину в начале и конце аудиофайла."""
    try:
        audio = AudioSegment.from_file(input_path)
        # Поиск неслышимых сегментов
        nonsilent_ranges = detect_nonsilent(audio,
                                            min_silence_len=min_silence_len,
                                            silence_thresh=silence_thresh)
        if nonsilent_ranges:
            # Определение начала и конца неслышимых сегментов
            start_trim = nonsilent_ranges[0][0]
            end_trim = nonsilent_ranges[-1][1]
            # Обрезка аудио
            trimmed_audio = audio[start_trim:end_trim]
            trimmed_audio.export(output_path, format="wav")
    except Exception as e:
        print(f"remove_silence: Ошибка обработки {input_path}: {e}")


def convert_to_16000hz(
        input_path: Path,
        output_path: Path
) -> None:
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000)  # Установка частоты дискретизации
        audio.export(output_path, format="wav")
    except Exception as e:
        print(f"convert_to_16000hz: Ошибка обработки {input_path}: {e}")


def padding_file(
        input_path: Path,
        output_path: Path,
        duration: float | int,
        noise_factor: float = 0.02  # Коэффициент шума
) -> bool:
    signal, sr = librosa.load(input_path, sr=None)
    target_length = int(duration * sr)  # Количество сэмплов для duration
    # Если аудиофайл короче 2.6 сек, дополняем тишиной
    if len(signal) == target_length:
        return True
    elif len(signal) > target_length:
        os.remove(output_path)
        print(f"Deleted file: {output_path}")
        return False
    pad_length = target_length - len(signal)
    signal = np.pad(signal, (0, pad_length), mode='constant')
    # Генерируем белый шум
    noise = np.random.randn(target_length)
    # Добавляем белый шум
    signal_noisy = signal + noise_factor * noise
    # Сохраняем зашумленный файл
    soundfile.write(output_path, signal_noisy, sr)
    return True


def transport_one_file(
    old_full_path: Path,
    new_full_path: Path,
) -> None:
    if old_full_path.suffix == ".ogg":
        convert_ogg_to_wav(old_full_path, new_full_path)
    elif old_full_path.suffix == ".opus":
        convert_opus_to_wav(old_full_path, new_full_path)
    elif old_full_path.suffix == ".mp3":
        convert_mp3_to_wav(old_full_path, new_full_path)
    elif old_full_path.suffix == ".wav":
        move_file(old_full_path, new_full_path)
    else:
        print(f"transport_files: пропуск файла {old_full_path}",
              file=sys.stderr)
        return
    convert_wav_to_16bit(new_full_path, new_full_path)
    convert_stereo_to_mono(new_full_path, new_full_path)
    remove_silence(new_full_path, new_full_path)
    convert_to_16000hz(new_full_path, new_full_path)
    padding_file(new_full_path, new_full_path, duration=2.6)
    print(new_full_path, flush=True)


def transport_files(
        old_dir: Path | str,
        new_dir: Path | str | None = None
) -> None:
    if new_dir and not path.exists(new_dir):
        makedirs(new_dir)
    for dirname, _, filenames in walk(old_dir):
        for filename in map(Path, filenames):
            old_full_path = Path(old_dir) / filename
            if new_dir is not None:
                new_full_path = (Path(new_dir) / filename).with_suffix(".wav")
            else:
                for pattern in (
                        "Catch", "Gun", "Index", "Like", "Relax", "Rock"
                ):
                    if pattern in str(old_full_path):
                        new_full_path = (
                                Path("output") / "ML" / pattern / filename)
                        new_full_path = new_full_path.with_suffix(".wav")
                        break
                else:
                    print(f"transport_files: пропуск файла {old_full_path}",
                          file=sys.stderr)
                    continue
            transport_one_file(old_full_path, new_full_path)


def squeeze(
        audio: tf.Tensor,
        labels: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    spectrogram = tf.signal.stft(
        waveform, frame_length=128, frame_step=64)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def make_spec_ds(ds: tf.Tensor) -> tf.Tensor:
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
