import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore
from clearml.task import TaskInstance  # type: ignore

from clearml import Task, TaskTypes  # type: ignore
from train.config import Config  # type: ignore
from validation.classes.revision_result import RevisionResult  # type: ignore
from validation.clearml_task_api import (  # type: ignore
    get_last_tasks,
    get_tasks_by_ids,
)
from validation.utils import evaluate_model  # type: ignore

warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive"
)


def validate(
        tasks: list[TaskInstance],
        current_task: Task
) -> None:
    SPEC_DS_DIR = Path("data/spec_ds")
    TEST_DS_PATH = SPEC_DS_DIR / "test_specs"
    LABELS_PATH = SPEC_DS_DIR / "label_names.npy"

    dummy = tf.data.Dataset.load(str(TEST_DS_PATH), compression="GZIP")
    spec = dummy.element_spec
    ds = tf.data.Dataset.load(str(TEST_DS_PATH),
                              element_spec=spec,
                              compression="GZIP")
    test_ds = ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    label_names = np.load(LABELS_PATH)

    results: dict[str, RevisionResult] = {}
    for i, task in enumerate(tasks):
        current_task.set_progress(int(i / len(tasks) * 100))

        rev = str(task.task_id)
        results[rev] = RevisionResult(label_names, task)

        model_files: list[Path] = []
        for art_name in task.artifacts.keys():
            if art_name.startswith("model"):
                model_files.append(Path(task.artifacts[art_name].get()))

        for j, filename in enumerate(model_files):
            current_task.set_progress(int(
                (i + j / len(model_files)) / len(tasks) * 100
            ))
            results[rev].add(*evaluate_model(filename, test_ds))

    with open("comparison_versions.md", "w", encoding='utf-8') as mf:
        mf.write("# Сравнение версий модели на тесте\n\n")
        for item in results.values():
            item.write_as_markdown(mf)


def main() -> None:
    task = Task.init(
        project_name=Config.PROJECT_NAME,
        task_name='validation',
        task_type=TaskTypes.testing,
        auto_connect_frameworks={
            'matplotlib': True,
            'tensorflow': False,
            'detect_repository': False,
        }
    )
    task.set_progress(0)
    parser = ArgumentParser(
        description="Validation tool for ML project with ClearML"
    )
    parser.add_argument('--project', '-p',
                        default=Config.PROJECT_NAME,
                        help="Project name")
    parser.add_argument('--task', '-t',
                        default='train model',
                        help="Task name")
    parser.add_argument('--number', '-n', type=int,
                        default=-1,
                        help="The last tasks to validate")
    parser.add_argument('--ids', nargs='*', type=str)
    args = parser.parse_args()
    if args.ids:
        tasks = get_tasks_by_ids(
            project_name=args.project,
            ids=args.ids,
        )
    else:
        tasks = get_last_tasks(
            project_name=args.project,
            task_name=args.task,
            count=args.number,
        )
    validate(tasks, task)
    task.set_progress(100)
    task.close()


if __name__ == '__main__':
    main()
