from dataclasses import dataclass


@dataclass
class Config:
    SEED: int = 42

    AUDIO_LENGTH: int = 6400

    SPECTROGRAM_WIDTH: int = 128
    SPECTROGRAM_HEIGHT: int = 64

    LEARNING_RATE: float = 0.001
    EPOCHS = 1000
    BATCH_SIZE = 64

    NUM_FOLDS = 5
