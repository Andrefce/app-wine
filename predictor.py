import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Optional

class WineQualityPredictor:
    """Handles predictions using different models"""
    def __init__(self, config):
        self.config = config
        self.X_scaler: Optional[StandardScaler] = None
        self.y_scaler: Optional[StandardScaler] = None
        self.linear_reg_model = None
        self.batch_theta: Optional[np.ndarray] = None
        self.stochastic_theta: Optional[np.ndarray] = None
        self.mini_batch_theta: Optional[np.ndarray] = None
        self.load_models_and_scalers()

    def load_models_and_scalers(self):
        """Load models and scalers from disk."""
        try:
            # Load scalers
            with open(Path(self.config.SCALER_DIR) / "X_scaler.pkl", 'rb') as f:
                self.X_scaler = pickle.load(f)
            with open(Path(self.config.SCALER_DIR) / "y_scaler.pkl", 'rb') as f:
                self.y_scaler = pickle.load(f)
            
            # Load models
            with open(Path(self.config.MODEL_DIR) / "linear_reg_model.pkl", 'rb') as f:
                self.linear_reg_model = pickle.load(f)
            with open(Path(self.config.MODEL_DIR) / "batch_theta.pkl", 'rb') as f:
                self.batch_theta = pickle.load(f)
            with open(Path(self.config.MODEL_DIR) / "stochastic_theta.pkl", 'rb') as f:
                self.stochastic_theta = pickle.load(f)
            with open(Path(self.config.MODEL_DIR) / "mini_batch_theta.pkl", 'rb') as f:
                self.mini_batch_theta = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load models or scalers: {e}")

    def predict_all(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """
        Make predictions using all models.
        
        Args:
            input_data: Dictionary of input features.
        
        Returns:
            Dictionary of predictions from all models.
        """
        try:
            # Convert input data to a DataFrame with the correct feature names
            input_df = pd.DataFrame([input_data])

            # Ensure the input DataFrame has the same feature names and order as the training data
            if hasattr(self.X_scaler, 'feature_names_in_'):
                input_df = input_df.reindex(columns=self.X_scaler.feature_names_in_, fill_value=0)

            # Convert to numpy array and scale features
            features_array = input_df.to_numpy()
            features_scaled = self.X_scaler.transform(features_array)
            features_scaled_with_bias = np.c_[np.ones((features_scaled.shape[0], 1)), features_scaled]

            # Make predictions
            predictions = {
                'batch': self._predict_with_theta(features_scaled_with_bias, self.batch_theta),
                'stochastic': self._predict_with_theta(features_scaled_with_bias, self.stochastic_theta),
                'mini_batch': self._predict_with_theta(features_scaled_with_bias, self.mini_batch_theta),
                'linear_regression': self._predict_linear_regression(features_scaled)
            }
            
            return predictions
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def _predict_with_theta(self, features: np.ndarray, theta: np.ndarray) -> float:
        """Predict using a theta vector (for batch, stochastic, and mini-batch models)."""
        prediction = features.dot(theta.T)
        return float(self.y_scaler.inverse_transform(prediction.reshape(1, -1))[0][0])

    def _predict_linear_regression(self, features: np.ndarray) -> float:
        """Predict using the linear regression model."""
        prediction = self.linear_reg_model.predict(features)
        return float(self.y_scaler.inverse_transform(prediction.reshape(1, -1))[0][0])