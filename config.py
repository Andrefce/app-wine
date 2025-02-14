# config.py
class ModelConfig:
    """Configuration for the machine learning models"""
    def __init__(
        self,
        FEATURE_COLUMNS: list[str] | None = None,
        TARGET_COLUMN: str = 'quality',
        TEST_SIZE: float = 0.15,
        RANDOM_STATE: int = 42,
        DATA_SEPARATOR: str = ';',
        MODEL_DIR: str = 'models',
        SCALER_DIR: str = 'scalers'
    ):
        self.FEATURE_COLUMNS = FEATURE_COLUMNS or [
            'fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
        self.TARGET_COLUMN = TARGET_COLUMN
        self.TEST_SIZE = TEST_SIZE
        self.RANDOM_STATE = RANDOM_STATE
        self.DATA_SEPARATOR = DATA_SEPARATOR
        self.MODEL_DIR = MODEL_DIR
        self.SCALER_DIR = SCALER_DIR