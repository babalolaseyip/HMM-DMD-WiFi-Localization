"""Tests for data loading and preprocessing."""
import pytest
import numpy as np
from src.data.data_loader import CrawdadDataLoader
from src.data.preprocessor import RSSIPreprocessor


def test_synthetic_data_generation():
    loader = CrawdadDataLoader()
    X, y, rooms = loader.create_synthetic_data(n_samples=100)
    assert X.shape == (100, 20, 14)
    assert y.shape == (100, 2)
    assert rooms.shape == (100,)
    assert np.all(X >= -90) and np.all(X <= -30)


def test_preprocessor_missing_values():
    X = np.array([[1.0, np.nan, 3.0], [0.0, 5.0, 6.0]])
    preprocessor = RSSIPreprocessor(missing_value=-100.0)
    X_clean = preprocessor.handle_missing_values(X)
    assert not np.any(np.isnan(X_clean))
    assert X_clean[0, 1] == -100.0
    assert X_clean[1, 0] == -100.0


def test_preprocessor_normalization():
    np.random.seed(42)
    X = np.random.randn(10, 20, 14)
    preprocessor = RSSIPreprocessor()
    X_normalized = preprocessor.normalize(X, fit=True)
    X_flat = X_normalized.reshape(10, -1)
    assert abs(np.mean(X_flat)) < 1e-8
    assert abs(np.std(X_flat) - 1.0) < 0.2


def test_preprocessor_smoothing():
    np.random.seed(42)
    X = np.random.randn(5, 20, 3)
    preprocessor = RSSIPreprocessor()
    X_smoothed = preprocessor.smooth_temporal(X, window_size=3)
    assert X_smoothed.shape == X.shape
    assert np.var(X_smoothed) <= np.var(X)
