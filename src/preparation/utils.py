import shutil
import subprocess
import wave
from os import makedirs, path, walk
from pathlib import Path

import numpy as np
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


def transport_files(
        old_dir: Path | str,
        new_dir: Path | str | None
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
                    print(f"transport_files: пропуск файла {old_full_path}")
                    return
            if filename.suffix == ".ogg":
                convert_ogg_to_wav(old_full_path, new_full_path)
            elif filename.suffix == ".opus":
                convert_opus_to_wav(old_full_path, new_full_path)
            elif filename.suffix == ".mp3":
                convert_mp3_to_wav(old_full_path, new_full_path)
            elif filename.suffix == ".wav":
                move_file(old_full_path, new_full_path)
            else:
                print(f"transport_files: пропуск файла {old_full_path}")
