# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """Configuration for the machine learning models"""
    FEATURE_COLUMNS: List[str] = field(default_factory=lambda: [
        'fixed acidity', 'volatile acidity', 'citric acid',
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ])
    TARGET_COLUMN: str = 'quality'
    TEST_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    DATA_SEPARATOR: str = ';'
    MODEL_DIR: str = 'models'
    SCALER_DIR: str = 'scalers'