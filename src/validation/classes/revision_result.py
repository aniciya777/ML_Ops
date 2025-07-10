import base64
import io
from io import TextIOWrapper
from typing import Sequence

import numpy as np
import seaborn as sns  # type: ignore
from clearml import Task  # type: ignore
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from validation.classes.one_class_result import OneClassResult  # type: ignore
from validation.clearml_task_api import get_url_for_task  # type: ignore


class RevisionResult:
    def __init__(
            self,
            classes: Sequence[str],
            task: Task,
    ) -> None:
        self._num_folds = 0
        self._classes = classes
        self._acc = 0.0
        self._precisions = {cls: 0.0 for cls in classes}
        self._recall = {cls: 0.0 for cls in classes}
        self._f1 = {cls: 0.0 for cls in classes}
        self._task = task
        self._cm: list[NDArray[np.int64]] = []

    @property
    def task(self) -> Task:
        return self._task

    @property
    def id(self) -> str:
        return self._task.task_id

    def add(
            self,
            report: dict,
            confusion_matrix: NDArray[np.int64]
    ) -> 'RevisionResult':
        self._num_folds += 1
        self._acc += report["accuracy"]
        for i, cls in enumerate(self._classes):
            s_i = str(i)
            if s_i not in report:
                continue
            self._precisions[cls] += report[s_i]["precision"]
            self._recall[cls] += report[s_i]["recall"]
            self._f1[cls] += report[s_i]["f1-score"]
        self._cm.append(confusion_matrix)
        return self

    @property
    def accuracy(self) -> float:
        return self._acc / self._num_folds

    @property
    def confusion_matrix(self) -> NDArray[np.int64]:
        cms = np.stack(self._cm)
        mean_cm = cms.mean(axis=0)
        row_sums = mean_cm.sum(axis=1, keepdims=True)
        norm_cm = mean_cm / row_sums
        return norm_cm

    def __getitem__(self, item: str) -> OneClassResult:
        return OneClassResult(
            self._precisions[item] / self._num_folds,
            self._recall[item] / self._num_folds,
            self._f1[item] / self._num_folds,
        )

    def write_as_markdown(self, file: TextIOWrapper) -> None:
        file.write(f"## ClearML task id `{self.id}`\n")
        file.write(f"Открыть по [ссылке]({get_url_for_task(self._task)})\n\n")

        for param, value in self._task.get_parameters_as_dict(cast=True) \
                .get("Args", dict()).items():
            file.write(f"- *{param}*: `{value}`\n")
        file.write("\n")

        file.write(f"**Accuracy**: `{self.accuracy:.4f}` \n\n")
        file.write("| Class | Precision | Recall | F1 score |\n")
        file.write("|:------|:---------:|:------:|:--------:|\n")
        for cls in self._classes:
            file.write(f"| **{cls}** |")
            file.write(f" `{self[cls].precision:.4f}` |")
            file.write(f" `{self[cls].recall:.4f}` |")
            file.write(f" `{self[cls].f1:.4f}` |\n")
        file.write("\n")
        img = self.draw_confusion_matrix(
            is_show=True,
            is_base64=True,
            title=f"Confusion matrix task_id={self.id}",
        )
        file.write(f"![Confusion matrix](data:image/png;base64,{img})\n\n")

    def draw_confusion_matrix(
        self,
        is_show: bool = True,
        is_base64: bool = False,
        title: str | None = None,
    ) -> str | None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(self.confusion_matrix,
                    xticklabels=self._classes,
                    yticklabels=self._classes,
                    annot=True,
                    fmt='.2f'
                    )
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        if is_show and not is_base64:
            plt.show()
        elif is_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            if is_show:
                plt.show()
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('ascii')
        return None
