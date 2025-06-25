import logging

from .config import Config


def scheduler(epoch: int, lr: float) -> float:
    if epoch == 0:
        return Config.LEARNING_RATE
    if epoch > 0 and epoch % 400 == 0:
        lr *= 0.5
        logging.info(f'Change lr to {lr}')
    return lr
