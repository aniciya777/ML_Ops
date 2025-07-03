import os
from dataclasses import dataclass


@dataclass
class Config:
    SEED: int = 42

    AUDIO_LENGTH: int = 41600

    SPECTROGRAM_WIDTH: int = 128
    SPECTROGRAM_HEIGHT: int = 64

    LEARNING_RATE: float = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32

    NUM_FOLDS = 5

    NEPTUNE = os.getenv("NEPTUNE")
    NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
