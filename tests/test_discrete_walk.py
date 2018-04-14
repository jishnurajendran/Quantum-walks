import pytest
import numpy as np
from quantumwalks import QuantumWalk

def test_quantum_walk_initialization():
    steps = 10
    qw = QuantumWalk(steps)
    assert qw.N == steps
    assert qw.P == 2 * steps + 1

def test_probability_conservation():
    steps = 5
    qw = QuantumWalk(steps)
    prob, _ = qw.simulate()
    assert np.isclose(np.sum(prob), 1.0, atol=1e-10)

def test_statistics_calculation():
    steps = 5
    qw = QuantumWalk(steps)
    prob, _ = qw.simulate()
    stats = qw.analyze_statistics(prob)

    assert "mean_position" in stats
    assert "standard_deviation" in stats
    assert "variance" in stats
    assert "max_probability" in stats
    assert "min_probability" in stats
