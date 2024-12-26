import pytest
import numpy as np

@pytest.fixture
def hadamard_coin():
    """Fixture for Hadamard coin operator"""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

@pytest.fixture
def balanced_coin():
    """Fixture for balanced coin operator"""
    return np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)

@pytest.fixture
def not_gate():
    """Fixture for NOT gate (X gate)"""
    return np.array([[0, 1], [1, 0]])

@pytest.fixture
def identity_coin():
    """Fixture for identity coin operator"""
    return np.array([[1, 0], [0, 1]])

@pytest.fixture
def basic_coin_schedule():
    """Fixture for basic alternating coin schedule"""
    return {
        "A": np.array([[1, 1], [1, -1]]) / np.sqrt(2),  # Hadamard
        "B": np.array([[0, 1], [1, 0]])                 # NOT gate
    }

@pytest.fixture
def custom_schedule_func():
    """Fixture for custom schedule function"""
    def schedule(step: int) -> str:
        if step < 2:
            return "A"
        elif step % 3 == 0:
            return "B"
        else:
            return "C"
    return schedule
