import numpy as np
import pandas as pd
from io import StringIO
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from config import ModelConfig

BLOB_URL = "https://winedataset.blob.core.windows.net/dataset/winequality-white.csv"

def load_dataset():
    """
    Download the dataset from Azure Blob Storage and load it into a Pandas DataFrame.
    """
    try:
        # Download the CSV file
        response = requests.get(BLOB_URL)
        response.raise_for_status()  # Raise an error for bad status codes

        # Load the CSV data into a Pandas DataFrame
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, delimiter=";")  # Adjust delimiter if needed
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

class DataProcessor:
    """Handles data loading and preprocessing"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def load_and_preprocess_data(self):
        try:
            df = load_dataset()
            X = df[self.config.FEATURE_COLUMNS]
            y = df[self.config.TARGET_COLUMN].values.reshape(-1, 1)
            
            X_scaled = self.X_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE
            )
            
            X_train_with_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
            X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
            
            # Save scalers
            Path(self.config.SCALER_DIR).mkdir(parents=True, exist_ok=True)
            with open(f"{self.config.SCALER_DIR}/X_scaler.pkl", 'wb') as f:
                pickle.dump(self.X_scaler, f)
            with open(f"{self.config.SCALER_DIR}/y_scaler.pkl", 'wb') as f:
                pickle.dump(self.y_scaler, f)
            
            return X_train_with_bias, X_test_with_bias, y_train, y_test
            
        except Exception as e:
            raise RuntimeError(f"Error in data preprocessing: {str(e)}")
