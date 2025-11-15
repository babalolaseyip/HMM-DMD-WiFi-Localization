"""Data loader for CRAWDAD WiFi fingerprinting dataset."""
import numpy as np
import pandas as pd
import os
from typing import Tuple


class CrawdadDataLoader:
    """Load and prepare CRAWDAD Mannheim/Compass dataset.

    Dataset: https://crawdad.org/mannheim/compass/20080411/802.11
    """

    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        self.radio_map = None

    def load_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        file_path = os.path.join(self.data_path, data_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}\nPlease download the CRAWDAD dataset and place it in {self.data_path}")
        data = pd.read_csv(file_path)
        rssi_cols = [col for col in data.columns if col.startswith('ap_')]
        if rssi_cols:
            X = data[rssi_cols].values
            if X.ndim == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            numeric_cols = data.select_dtypes(include='number').columns.tolist()
            X = data[numeric_cols].values
            X = X.reshape(X.shape[0], 1, X.shape[1])
        if {'x', 'y'}.issubset(set(data.columns)):
            y = data[['x', 'y']].values
        else:
            y = np.zeros((len(data), 2))
        if 'room' in data.columns:
            rooms = data['room'].values
        else:
            rooms = np.zeros(len(data))
        return X, y, rooms

    def load_training_data(self):
        return self.load_data('training_data.csv')

    def load_test_data(self):
        return self.load_data('test_data.csv')

    def create_synthetic_data(self, n_samples: int = 1000, n_timestamps: int = 20, n_aps: int = 14):
        np.random.seed(42)
        X = -90 + 60 * np.random.rand(n_samples, n_timestamps, n_aps)
        y = 50 * np.random.rand(n_samples, 2)
        rooms = np.random.randint(0, 7, size=n_samples)
        return X, y, rooms
