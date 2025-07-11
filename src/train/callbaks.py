from datetime import datetime

from clearml import Task  # type: ignore
from tensorflow.keras.callbacks import Callback  # type: ignore


class ClearMLLogger(Callback):
    def __init__(
            self,
            task: Task,
            fold_index: int,
            folds_count: int,
            count_epochs: int
    ) -> None:
        super().__init__()
        self.task: Task = task
        self.logger = task.get_logger()
        self.fold_index: int = fold_index
        self.count_epochs: int = count_epochs
        self.folds_count: int = folds_count

        self._last_time: datetime | None = None

    def on_epoch_end(self, epoch: int, logs=None):
        for name, value in (logs or {}).items():
            self.logger.report_scalar(
                title=name,
                series=f"#{self.fold_index + 1} Fold",
                value=value,
                iteration=epoch
            )
        self.task.set_progress(int(
            100 * (self.fold_index + epoch / self.count_epochs)
            / self.folds_count
        ))
        current_datetime = datetime.now()
        if self._last_time is not None:
            self.logger.report_scalar(
                title="Throughput, microseconds",
                series=f"#{self.fold_index + 1} Fold",
                value=(current_datetime - self._last_time).microseconds,
                iteration=epoch
            )
        self._last_time = current_datetime
