import shutil
import subprocess
import wave
from os import PathLike, makedirs, path

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


# Функция для конвертации ogg в wav
def convert_ogg_to_wav(
        input_file: PathLike[str],
        output_file: PathLike[str]
        ) -> None:
    try:
        # Загружаем ogg файл
        audio = AudioSegment.from_file(input_file, format="ogg")

        # Сохраняем как wav
        audio.export(output_file, format="wav")
    except Exception as e:
        print(f"convert_ogg_to_wav: Ошибка при конвертации: {e}")


def convert_mp3_to_wav(
        input_file: PathLike[str],
        output_file: PathLike[str]
        ) -> None:
    try:
        # Загружаем mp3 файл
        audio = AudioSegment.from_mp3(input_file)
        # Сохраняем как wav
        audio.export(output_file, format="wav")
    except Exception as e:
        print(f"convert_mp3_to_wav: Ошибка при конвертации: {e}")


def convert_opus_to_wav(
        input_file: PathLike[str],
        output_file: PathLike[str]
        ) -> None:
    try:
        if not path.exists(input_file):
            print(f"convert_opus_to_wav: Файл {input_file} не найден!")
            return
        command = [
            "ffmpeg",
            "-i", str(input_file),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_file)
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


def move_file(source: PathLike[str], destination: PathLike[str]) -> None:
    try:
        shutil.copy2(source, destination)
    except Exception as e:
        print(f"move_file: Ошибка при перемещении файла: {e}")


def convert_wav_to_16bit(
        input_file: PathLike[str],
        output_file: PathLike[str]
        ) -> None:
    try:
        # Загружаем WAV файл
        audio = AudioSegment.from_wav(input_file)
        # Преобразуем в 16-битный формат
        audio = audio.set_sample_width(2)  # 2 байта = 16 бит
        # Сохраняем как 16-битный WAV
        audio.export(output_file, format="wav")
    except Exception as e:
        print(f"convert_wav_to_16bit: Ошибка при конвертации файла {input_file}: {e}")


def convert_stereo_to_mono(
        input_file: PathLike[str],
        output_file: PathLike[str]
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
        print(f"convert_stereo_to_mono: Ошибка при конвертации файла {input_file}: {e}")