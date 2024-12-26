import pytest
import numpy as np
from quantumwalks import QuantumWalk
from quantumwalks.utils.operators import CoinSchedule

def test_basic_quantum_walk(hadamard_coin):
    """Test basic quantum walk with default parameters"""
    steps = [1, 2, 5]
    for num_steps in steps:
        qw = QuantumWalk(num_steps=num_steps)
        prob, _ = qw.simulate()

        # Test probability conservation
        assert np.allclose(np.sum(prob), 1.0)

        # Test probability is real and non-negative
        assert np.all(prob >= 0)
        assert np.all(np.isreal(prob))

        # Test distribution size
        assert len(prob) == 2 * num_steps + 1

def test_coin_operators(hadamard_coin, balanced_coin, not_gate, identity_coin):
    """Test quantum walk with different coin operators"""
    coins = [hadamard_coin, balanced_coin, not_gate, identity_coin]

    for coin in coins:
        qw = QuantumWalk(num_steps=2, coin_schedule=coin)
        prob, _ = qw.simulate()

        assert np.allclose(np.sum(prob), 1.0)
        assert np.all(prob >= 0)

def test_initial_states():
    """Test quantum walk with different initial states"""
    num_steps = 2

    # Test different initial coin states
    init_states = [
        np.array([1, 0]),              # |0⟩
        np.array([0, 1]),              # |1⟩
        np.array([1, 1]) / np.sqrt(2), # |+⟩
        np.array([1, -1]) / np.sqrt(2) # |-⟩
    ]

    for state in init_states:
        qw = QuantumWalk(num_steps=num_steps, initial_coin_state=state)
        prob, _ = qw.simulate()
        assert np.allclose(np.sum(prob), 1.0)

    # Test different initial positions
    positions = [0, 1, 2]
    for pos in positions:
        qw = QuantumWalk(num_steps=num_steps, initial_position=pos)
        prob, _ = qw.simulate()
        assert np.allclose(np.sum(prob), 1.0)

def test_dynamic_coins(basic_coin_schedule, custom_schedule_func):
    """Test quantum walk with dynamic coin operators"""
    num_steps = 4

    # Test alternating coins
    qw = QuantumWalk(num_steps=num_steps, coin_schedule=basic_coin_schedule)
    prob, _ = qw.simulate()
    assert np.allclose(np.sum(prob), 1.0)

    # Test custom coin schedule
    custom_coins = {
        "A": basic_coin_schedule["A"],
        "B": basic_coin_schedule["B"],
        "C": np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    }

    schedule = CoinSchedule(
        schedule_type="custom",
        coins=custom_coins,
        schedule_func=custom_schedule_func
    )

    qw = QuantumWalk(num_steps=num_steps, coin_schedule=schedule)
    prob, _ = qw.simulate()
    assert np.allclose(np.sum(prob), 1.0)

def test_state_evolution():
    """Test quantum walk state evolution"""
    qw = QuantumWalk(num_steps=3)
    prob, _, states = qw.simulate(store_states=True)

    # Test number of stored states
    assert len(states) == 4  # initial + 3 steps

    # Test each state is normalized
    for state in states:
        assert np.allclose(np.sum(np.abs(state)**2), 1.0)

    # Test state evolution is unitary
    for i in range(len(states)-1):
        norm_diff = np.sum(np.abs(states[i])**2) - np.sum(np.abs(states[i+1])**2)
        assert np.allclose(norm_diff, 0.0)

def test_statistics():
    """Test statistical analysis of quantum walk"""
    qw = QuantumWalk(num_steps=10)
    prob, _ = qw.simulate()
    stats = qw.analyze_statistics(prob)

    required_stats = [
        "mean_position",
        "standard_deviation",
        "variance",
        "max_probability",
        "min_probability",
        "entropy"
    ]

    # Test presence of all statistics
    for stat in required_stats:
        assert stat in stats

    # Test validity of statistics
    assert stats["standard_deviation"] >= 0
    assert stats["variance"] >= 0
    assert 0 <= stats["max_probability"] <= 1
    assert 0 <= stats["min_probability"] <= stats["max_probability"]

def test_invalid_parameters():
    """Test error handling for invalid parameters"""
    invalid_cases = [
        (0, ValueError),                    # invalid steps
        (-1, ValueError),                   # negative steps
        (None, TypeError)                   # wrong type
    ]

    for steps, error_type in invalid_cases:
        with pytest.raises(error_type):
            QuantumWalk(num_steps=steps)

@pytest.mark.parametrize("steps", [1, 2, 5, 10])
def test_walk_scaling(steps):
    """Test quantum walk behavior with different numbers of steps"""
    qw = QuantumWalk(num_steps=steps)
    prob, _ = qw.simulate()

    # Test probability conservation
    assert np.allclose(np.sum(prob), 1.0)
    assert len(prob) == 2 * steps + 1

    # Calculate distribution properties
    positions = np.arange(-steps, steps + 1)
    mean_pos = np.average(positions, weights=prob)
    variance = np.average((positions - mean_pos)**2, weights=prob)

    # For quantum walks, we expect:
    # 1. Symmetry around initial position (mean ≈ 0)
    assert np.abs(mean_pos) < 1e-10

    # 2. Variance should scale with steps^2
    # For small numbers of steps, just verify it's positive
    assert variance >= 0

    # For larger numbers of steps, can check scaling
    if steps > 5:
        classical_variance = steps  # classical random walk
        assert variance > classical_variance  # quantum walk spreads faster
