import os

import matplotlib.pyplot as plt
from pydub import AudioSegment


def get_audio_duration(filepath) -> float | None:
    """Возвращает длительность аудиофайла в секундах."""
    try:
        audio = AudioSegment.from_file(
            filepath)  # pydub автоматически определяет формат
        return len(audio) / 1000  # Длительность в секундах
    except Exception as e:
        print(f"Не удалось прочитать файл {filepath}: {e}")
        return None


def main() -> None:
    # Корневая директория
    root_dir = "../../data/output"  # Укажите свою директорию

    audio_durations = []

    # Рекурсивный обход папок
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".wav", ".opus", ".ogg", ".mp3",
                                      ".flac")):  # Добавляем больше форматов
                file_path = os.path.join(root, file)
                duration = get_audio_duration(file_path)
                if duration is not None:
                    audio_durations.append(duration)

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.hist(audio_durations, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Длительность аудиофайла (секунды)")
    plt.ylabel("Количество файлов")
    plt.title("Распределение длительности аудиофайлов")
    plt.show()


if __name__ == '__main__':
    main()
