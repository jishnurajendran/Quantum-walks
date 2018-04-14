import numpy as np

def create_hadamard() -> np.ndarray:
    """Create Hadamard operator"""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def create_shift_operators(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create shift operators"""
    shift_plus = np.roll(np.eye(size), 1, axis=0)
    shift_minus = np.roll(np.eye(size), -1, axis=0)
    return shift_plus, shift_minus
