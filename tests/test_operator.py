import pytest
import numpy as np
from scipy.sparse import csc_matrix
from quantumwalks.utils.operators import (
    create_hadamard, create_custom_coin,
    create_sparse_shift_operators, create_initial_state,
    CoinSchedule
)

def test_hadamard_properties(hadamard_coin):
    """Test basic properties of Hadamard operator"""
    H = create_hadamard()
    H_dense = H.toarray()

    # Test unitarity
    assert np.allclose(H_dense @ H_dense.conj().T, np.eye(2))

    # Test hermiticity
    assert np.allclose(H_dense, H_dense.conj().T)

    # Test correct form
    assert np.allclose(H_dense, hadamard_coin)

def test_hadamard_normalization():
    """Test normalization preservation of Hadamard operator"""
    H = create_hadamard()
    basis_states = [
        np.array([1, 0]),  # |0⟩
        np.array([0, 1]),  # |1⟩
        np.array([1, 1])/np.sqrt(2)  # |+⟩
    ]

    for state in basis_states:
        evolved = H.dot(state)
        assert np.isclose(np.linalg.norm(evolved), 1.0)

def test_custom_coin_validation(balanced_coin, not_gate, identity_coin):
    """Test custom coin operator validation"""
    # Test valid coins
    valid_coins = [balanced_coin, not_gate, identity_coin]
    for coin in valid_coins:
        assert isinstance(create_custom_coin(coin), csc_matrix)

    # Test invalid shapes
    invalid_shapes = [
        np.array([[1]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([1, 1])
    ]
    for invalid_coin in invalid_shapes:
        with pytest.raises(ValueError):
            create_custom_coin(invalid_coin)

    # Test non-unitary matrices
    non_unitary = [
        np.array([[1, 1], [1, 1]]),
        np.array([[2, 0], [0, 2]]),
        np.array([[1, 2], [3, 4]])
    ]
    for invalid_coin in non_unitary:
        with pytest.raises(ValueError):
            create_custom_coin(invalid_coin)

def test_shift_operators():
    """Test shift operators properties"""
    sizes = [3, 5, 7]
    for size in sizes:
        shift_plus, shift_minus = create_sparse_shift_operators(size)

        # Test shapes
        assert shift_plus.shape == (size, size)
        assert shift_minus.shape == (size, size)

        # Test periodic boundary conditions
        state = np.zeros(size)
        state[0] = 1

        shifted_right = shift_plus.dot(state)
        assert np.isclose(shifted_right[1], 1.0)

        shifted_left = shift_minus.dot(state)
        assert np.isclose(shifted_left[-1], 1.0)

def test_custom_shift_operators():
    """Test custom shift operators"""
    def custom_shift(size):
        shift_plus = np.roll(np.eye(size), 1, axis=0)
        shift_minus = np.roll(np.eye(size), -1, axis=0)
        return shift_plus, shift_minus

    sizes = [3, 5, 7]
    for size in sizes:
        shift_plus, shift_minus = create_sparse_shift_operators(size, custom_shift)

        assert isinstance(shift_plus, csc_matrix)
        assert isinstance(shift_minus, csc_matrix)
        assert shift_plus.shape == (size, size)

        # Test unitarity
        shift_plus_dense = shift_plus.toarray()
        assert np.allclose(shift_plus_dense @ shift_plus_dense.T, np.eye(size))

def test_initial_state_creation():
    """Test initial state creation"""
    sizes = [3, 5, 7]
    for size in sizes:
        # Test default state
        posn0, coin0 = create_initial_state(size)
        assert isinstance(posn0, csc_matrix)
        assert isinstance(coin0, csc_matrix)

        # Test normalization
        assert np.isclose(np.sum(np.abs(posn0.toarray())**2), 1.0)
        assert np.isclose(np.sum(np.abs(coin0.toarray())**2), 1.0)

        # Test custom position
        for pos in range(size):
            posn0, _ = create_initial_state(size, position=pos)
            assert np.isclose(posn0.toarray()[pos, 0], 1.0)

        # Test custom coin state
        custom_coins = [
            np.array([1, 0]),              # |0⟩
            np.array([0, 1]),              # |1⟩
            np.array([1, 1])/np.sqrt(2),   # |+⟩
            np.array([1, -1])/np.sqrt(2),  # |-⟩
            np.array([1, 1j])/np.sqrt(2)   # |+i⟩
        ]
        for coin_state in custom_coins:
            _, coin0 = create_initial_state(size, coin_state=coin_state)
            assert np.allclose(coin0.toarray().flatten(), coin_state)

def test_coin_schedule_creation(basic_coin_schedule, custom_schedule_func):
    """Test CoinSchedule initialization and validation"""
    # Test fixed schedule
    fixed_schedule = CoinSchedule("fixed", coins={"default": basic_coin_schedule["A"]})
    assert isinstance(fixed_schedule.get_coin(0), csc_matrix)

    # Test alternating schedule
    alt_schedule = CoinSchedule("alternating", coins=basic_coin_schedule)
    assert isinstance(alt_schedule.get_coin(0), csc_matrix)
    assert isinstance(alt_schedule.get_coin(1), csc_matrix)

    # Test custom schedule
    custom_coins = {
        "A": basic_coin_schedule["A"],
        "B": basic_coin_schedule["B"],
        "C": np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    }
    custom_schedule = CoinSchedule("custom", coins=custom_coins, schedule_func=custom_schedule_func)
    assert isinstance(custom_schedule.get_coin(0), csc_matrix)

    # Test invalid schedule type (should raise ValueError)
    with pytest.raises(ValueError, match="Unknown schedule type"):
        invalid_schedule = CoinSchedule("invalid_type", coins=basic_coin_schedule)
        invalid_schedule.get_coin(0)  # This should raise the error

def test_coin_schedule_patterns(basic_coin_schedule, custom_schedule_func):
    """Test coin schedule patterns"""
    # Test alternating pattern
    alt_schedule = CoinSchedule("alternating", coins=basic_coin_schedule)
    steps = [0, 1, 2, 3, 4]
    expected_pattern = ["A", "B", "A", "B", "A"]

    for step, expected in zip(steps, expected_pattern):
        coin = alt_schedule.get_coin(step).toarray()
        assert np.allclose(coin, basic_coin_schedule[expected])

    # Test custom pattern
    custom_coins = {
        "A": basic_coin_schedule["A"],
        "B": basic_coin_schedule["B"],
        "C": np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    }
    custom_schedule = CoinSchedule("custom", coins=custom_coins, schedule_func=custom_schedule_func)

    expected_pattern = ["A", "A", "C", "B", "C"]
    for step, expected in zip(steps, expected_pattern):
        coin = custom_schedule.get_coin(step).toarray()
        assert np.allclose(coin, custom_coins[expected])
