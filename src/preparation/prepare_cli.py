import logging
import os.path
import sys
from pathlib import Path

from clearml import Task  # type: ignore

from utils import transport_one_file  # type: ignore


def main() -> None:
    task = Task.init(project_name='HAND', task_name='prepare audio')
    logger = task.get_logger()
    files = list(map(str.strip, sys.stdin))
    for i, file in enumerate(files):
        task.set_progress(int(i / len(files) * 100))
        try:
            inp = Path(file)
            assert os.path.exists(inp), f'{inp} does not exist'
            out = Path(
                file
                .replace('/Whatsapp', '')
                .replace("data/input/ML", "data/output/ML")
                + '.wav'
            )
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
