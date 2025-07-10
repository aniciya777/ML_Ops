from typing import Sequence

from clearml.task import Task, TaskInstance  # type: ignore


def get_last_tasks(
        project_name: str,
        task_name: str,
        count: int
) -> list[TaskInstance]:
    tasks = Task.get_tasks(
        project_name=project_name,
        task_name=task_name,
        task_filter={'order_by': ['-last_update']}
    )[::-1]
    if count != -1:
        tasks = tasks[-count:]
    return tasks


def get_tasks_by_ids(
        project_name: str,
        ids: Sequence[str]
) -> list[TaskInstance]:
    return Task.get_tasks(
        task_ids=ids,
        project_name=project_name
    )


def get_url_for_task(task: Task) -> str:
    return Task.get_task_output_log_web_page(
        task_id=task.task_id,
        project_id=task.project
    )
