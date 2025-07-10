from typing import NamedTuple


class OneClassResult(NamedTuple):
    precision: float
    recall: float
    f1: float
