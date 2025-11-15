"""Tests for feature extractors."""
import pytest
import numpy as np
from src.features.dmd_extractor import DMDExtractor


def test_dmd_extractor_init():
    extractor = DMDExtractor(rank=7)
    assert extractor.rank == 7


def test_dmd_extractor_transform():
    np.random.seed(42)
    X = np.random.randn(10, 20, 14)
    extractor = DMDExtractor(rank=7)
    features = extractor.transform(X)
    assert features.shape == (10, 7)
    assert not np.any(np.isnan(features))


def test_dmd_modes_computation():
    np.random.seed(42)
    X = np.random.randn(20, 14)
    extractor = DMDExtractor(rank=7)
    dmd_modes = extractor._compute_dmd_modes(X)
    assert dmd_modes.shape[1] == 7
    assert not np.any(np.isnan(dmd_modes))
