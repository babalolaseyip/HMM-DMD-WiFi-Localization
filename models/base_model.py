"""Base class for localization models."""
from abc import ABC, abstractmethod
import numpy as np


class BaseLocalizationModel(ABC):
    """Abstract base class for indoor localization models."""

    @abstractmethod
    def fit(self, X, y, rooms=None):
        """Train the model.

        Args:
            X: RSSI measurements (n_samples, n_timestamps, n_aps)
            y: Ground truth coordinates (n_samples, 2)
            rooms: Room labels (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict locations.

        Args:
            X: RSSI measurements (n_samples, n_timestamps, n_aps)

        Returns:
            Predicted coordinates (n_samples, 2)
        """
        pass

    def score(self, X, y):
        """Calculate mean localization error."""
        predictions = self.predict(X)
        errors = np.linalg.norm(predictions - y, axis=1)
        return np.mean(errors)
