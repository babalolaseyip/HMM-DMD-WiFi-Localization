"""RSSI data preprocessing utilities."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class RSSIPreprocessor:
    """Preprocess RSSI measurements for WiFi localization."""

    def __init__(self, missing_value: float = -100.0):
        self.missing_value = missing_value
        self.scaler = StandardScaler()
        self.fitted = False

    def handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        X_clean = X.copy().astype(float)
        X_clean[np.isnan(X_clean)] = self.missing_value
        X_clean[X_clean == 0] = self.missing_value
        return X_clean

    def normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        if fit:
            X_normalized = self.scaler.fit_transform(X_flat)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_normalized = self.scaler.transform(X_flat)
        return X_normalized.reshape(original_shape)

    def smooth_temporal(self, X: np.ndarray, window_size: int = 3) -> np.ndarray:
        if len(X.shape) != 3:
            raise ValueError("Expected 3D array (n_samples, n_timestamps, n_aps)")
        X_smoothed = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                X_smoothed[i, :, j] = np.convolve(
                    X[i, :, j],
                    np.ones(window_size) / window_size,
                    mode='same'
                )
        return X_smoothed

    def preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = self.handle_missing_values(X)
        X = self.smooth_temporal(X)
        X = self.normalize(X, fit=fit)
        return X
