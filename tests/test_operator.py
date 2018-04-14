import pytest
import numpy as np
from quantumwalks.utils.operators import create_hadamard

def test_hadamard_properties():
    H = create_hadamard()
    # Test unitarity
    assert np.allclose(H.dot(H.T), np.eye(2))
    # Test hermiticity
    assert np.allclose(H, H.T)

def test_hadamard_normalization():
    H = create_hadamard()
    assert np.isclose(np.linalg.norm(H.dot(np.array([1, 0]))), 1.0)
    assert np.isclose(np.linalg.norm(H.dot(np.array([0, 1]))), 1.0)
