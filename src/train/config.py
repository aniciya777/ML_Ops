import os
from dataclasses import dataclass


@dataclass
class Config:
    PROJECT_NAME = 'HAND'

    SEED: int = 42

    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_DURATION: float = 2.6
    AUDIO_LENGTH: int = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)

    SPECTROGRAM_WIDTH: int = 128
    SPECTROGRAM_HEIGHT: int = 64

    LEARNING_RATE: float = 0.001
    EPOCHS = 1000
    BATCH_SIZE = 32

    NUM_FOLDS = 5
