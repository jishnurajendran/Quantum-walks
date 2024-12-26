import numpy as np
from scipy.sparse import csc_matrix, eye
from typing import Tuple, Optional, Union, Callable, Dict
from functools import lru_cache

class CoinSchedule:
    def __init__(self, schedule_type: str = "fixed",
                 coins: Dict[str, np.ndarray] = None,
                 schedule_func: Callable[[int], str] = None):
        """Initialize coin schedule"""
        if schedule_type not in ["fixed", "alternating", "custom"]:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.schedule_type = schedule_type
        self.coins = {}

        # Convert and validate all coins
        if coins:
            for name, coin in coins.items():
                self.coins[name] = create_custom_coin(coin)

        self.schedule_func = schedule_func

        # Validate required coins for each schedule type
        if schedule_type == "fixed" and "default" not in coins:
            raise ValueError("Fixed schedule requires 'default' coin")
        elif schedule_type == "alternating" and not all(k in coins for k in ["A", "B"]):
            raise ValueError("Alternating schedule requires coins 'A' and 'B'")
        elif schedule_type == "custom" and not schedule_func:
            raise ValueError("Custom schedule requires schedule_func")

    def get_coin(self, step: int) -> csc_matrix:
        """Get coin operator for given step"""
        if self.schedule_type == "fixed":
            return self.coins["default"]
        elif self.schedule_type == "alternating":
            return self.coins["A" if step % 2 == 0 else "B"]
        elif self.schedule_type == "custom":
            coin_name = self.schedule_func(step)
            if coin_name not in self.coins:
                raise ValueError(f"Invalid coin name: {coin_name}")
            return self.coins[coin_name]
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

def create_hadamard() -> csc_matrix:
    """
    Create Hadamard operator as sparse matrix

    Returns:
        csc_matrix: Sparse Hadamard operator
    """
    return csc_matrix(np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2))

def create_custom_coin(matrix: np.ndarray) -> csc_matrix:
    """
    Create a custom coin operator from a 2x2 matrix

    Args:
        matrix (np.ndarray): 2x2 unitary matrix

    Returns:
        csc_matrix: Sparse matrix representation of the coin operator

    Raises:
        ValueError: If matrix is not 2x2 or not unitary
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise ValueError("Coin operator must be a 2x2 matrix")

    # Precise unitarity check
    identity = np.eye(2)
    if not np.allclose(matrix @ matrix.conj().T, identity, atol=1e-10) or \
       not np.allclose(matrix.conj().T @ matrix, identity, atol=1e-10):
        raise ValueError("Coin operator must be unitary")

    return csc_matrix(matrix)

@lru_cache(maxsize=32)
def create_sparse_shift_operators(size: int,
                                custom_shift: Optional[Callable[[int], Tuple[np.ndarray, np.ndarray]]] = None
                                ) -> Tuple[csc_matrix, csc_matrix]:
    """
    Create sparse shift operators with optional custom definition

    Args:
        size (int): Size of the position space
        custom_shift (Optional[Callable]): Custom function that returns shift operator matrices

    Returns:
        Tuple[csc_matrix, csc_matrix]: Shift operators for right and left movement

    Raises:
        ValueError: If custom shift operators are invalid
    """
    if custom_shift is not None:
        shift_plus_matrix, shift_minus_matrix = custom_shift(size)

        # Validate custom shift operators
        for name, matrix in [("shift_plus", shift_plus_matrix), ("shift_minus", shift_minus_matrix)]:
            if matrix.shape != (size, size):
                raise ValueError(f"Custom {name} must be {size}x{size}")

        return (csc_matrix(shift_plus_matrix), csc_matrix(shift_minus_matrix))

    # Efficient default shift operators construction
    data = np.ones(size, dtype=np.complex128)
    indices = np.arange(size)
    indptr = np.arange(size + 1)

    indices_plus = (indices + 1) % size
    indices_minus = (indices - 1) % size

    shift_plus = csc_matrix((data, indices_plus, indptr), shape=(size, size))
    shift_minus = csc_matrix((data, indices_minus, indptr), shape=(size, size))

    return shift_plus, shift_minus

def create_initial_state(size: int,
                        position: Optional[Union[int, np.ndarray]] = None,
                        coin_state: Optional[np.ndarray] = None) -> Tuple[csc_matrix, csc_matrix]:
    """
    Create initial position and coin states

    Args:
        size (int): Size of the position space
        position (Optional[Union[int, np.ndarray]]): Initial position or distribution
        coin_state (Optional[np.ndarray]): Initial coin state

    Returns:
        Tuple[csc_matrix, csc_matrix]: Initial position and coin states

    Raises:
        ValueError: If input states are invalid
    """
    # Handle position state
    if position is None:
        posn0 = csc_matrix(([1], ([size//2], [0])), shape=(size, 1))
    elif isinstance(position, (int, np.integer)):
        if not 0 <= position < size:
            raise ValueError(f"Position must be between 0 and {size-1}")
        posn0 = csc_matrix(([1], ([position], [0])), shape=(size, 1))
    else:
        position = np.asarray(position, dtype=np.complex128)
        if position.shape != (size,):
            raise ValueError(f"Position distribution must have length {size}")
        if not np.allclose(np.sum(np.abs(position)**2), 1, atol=1e-10):
            raise ValueError("Position distribution must be normalized")
        posn0 = csc_matrix(position.reshape(-1, 1))

    # Handle coin state
    if coin_state is None:
        coin0 = (np.array([1, -1j], dtype=np.complex128)/np.sqrt(2)).reshape(-1, 1)
    else:
        coin_state = np.asarray(coin_state, dtype=np.complex128)
        if coin_state.shape != (2,):
            raise ValueError("Coin state must be a 2D vector")
        if not np.allclose(np.linalg.norm(coin_state), 1, atol=1e-10):
            raise ValueError("Coin state must be normalized")
        coin0 = coin_state.reshape(-1, 1)

    return posn0, csc_matrix(coin0)

def create_parametric_coin(alpha: float, beta: float, gamma: float) -> csc_matrix:
    """
    Create a parametric coin operator U(α,β,γ).

    Args:
        alpha (float): Rotation angle α
        beta (float): Rotation angle β
        gamma (float): Rotation angle γ

    Returns:
        csc_matrix: Parametric coin operator

    Raises:
        ValueError: If resulting operator is not unitary
    """
    # Create the coin operator
    coin = np.array([
        [np.exp(1j * alpha) * np.cos(beta), -np.exp(-1j * gamma) * np.sin(beta)],
        [np.exp(1j * gamma) * np.sin(beta), np.exp(-1j * alpha) * np.cos(beta)]
    ], dtype=np.complex128)

    # Verify unitarity
    identity = np.eye(2)
    if not np.allclose(coin @ coin.conj().T, identity, atol=1e-10) or \
       not np.allclose(coin.conj().T @ coin, identity, atol=1e-10):
        raise ValueError("Resulting coin operator is not unitary")

    return csc_matrix(coin)
