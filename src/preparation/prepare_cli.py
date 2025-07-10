import logging
import os.path
import sys
from pathlib import Path

from clearml import Task, TaskTypes  # type: ignore
from matplotlib import pyplot as plt
from utils import get_audio_duration, transport_one_file  # type: ignore

from train.config import Config  # type: ignore


def main() -> None:
    task = Task.init(
        project_name=Config.PROJECT_NAME,
        task_name='prepare audio',
        task_type=TaskTypes.data_processing
    )
    logger = task.get_logger()
    logger.report_single_value("sample rate", Config.AUDIO_SAMPLE_RATE)
    logger.report_single_value("duration", Config.AUDIO_DURATION)
    logger.report_single_value("length", Config.AUDIO_LENGTH)
    files = [Path(s.strip()) for s in sys.stdin]
    logger.report_single_value("count files", len(files))

    audio_durations = [
        get_audio_duration(file) for file in files
    ]
    plt.figure(figsize=(10, 6))
    plt.hist(
        audio_durations,
        bins=int(len(audio_durations) ** 0.5),
        color='skyblue',
        edgecolor='black'
    )
    plt.axvline(x=Config.AUDIO_DURATION, linestyle='--', linewidth=2)
    plt.xlabel("Длительность аудиофайла (секунды)")
    plt.ylabel("Количество файлов")
    plt.title("Распределение длительности аудиофайлов")
    plt.show()

    for i, inp in enumerate(files):
        task.set_progress(int(i / len(files) * 100))
        try:
            out = Path(
                str(inp)
                .replace('/Whatsapp', '')
                .replace("data/input/ML", "data/output/ML")
                + '.wav'
            )
            assert os.path.exists(inp), f'{inp} does not exist'
            out.parent.mkdir(parents=True, exist_ok=True)
            transport_one_file(inp, out)
            logger.report_text(f"✔️ Обработан файл: {inp} -> {out}")
        except Exception as e:
            print(e, file=sys.stderr)
            logger.report_text(f"Ошибка обработки файла {inp} -> {out}: {e}",
                               level=logging.ERROR)
    task.set_progress(100)
    task.close()


if __name__ == "__main__":
    main()
