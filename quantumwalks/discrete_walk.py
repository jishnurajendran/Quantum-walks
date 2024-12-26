import numpy as np
import time
from typing import Tuple, Optional, Callable, Union, List, Dict
from tqdm import tqdm
from scipy.sparse import kron, csc_matrix
from functools import lru_cache
from .utils.operators import (
    create_hadamard, create_sparse_shift_operators,
    create_custom_coin, create_initial_state, CoinSchedule
)
from .utils.visualization import plot_distribution

class QuantumWalk:
    """
    Quantum Walk implementation with customizable components and optimization.
    Implements a discrete-time quantum walk on a line with customizable coin and shift operators.
    """

    def __init__(self,
                 num_steps: int,
                 initial_position: Optional[Union[int, np.ndarray]] = None,
                 initial_coin_state: Optional[np.ndarray] = None,
                 coin_schedule: Optional[Union[np.ndarray, Dict[str, np.ndarray], CoinSchedule]] = None,
                 custom_shift: Optional[Callable[[int], Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Initialize Quantum Walk with customizable components.

        Args:
            num_steps (int): Number of steps for the walk
            initial_position (Optional[Union[int, np.ndarray]]): Initial position or distribution
            initial_coin_state (Optional[np.ndarray]): Initial coin state
            coin_schedule (Optional[Union[np.ndarray, Dict, CoinSchedule]]):
                Either a fixed coin operator, dictionary of coins, or CoinSchedule object
            custom_shift (Optional[Callable]): Custom shift operator function

        Raises:
            ValueError: If any input parameters are invalid
        """
        if num_steps < 1:
            raise ValueError("Number of steps must be positive")

        self.N = num_steps
        self.P = 2 * num_steps + 1

        # Create initial states
        self.posn0, self.initial_coin_state = create_initial_state(
            self.P, initial_position, initial_coin_state
        )

        # Set up coin basis states
        self.coin0 = csc_matrix(np.array([[1], [0]], dtype=np.complex128))
        self.coin1 = csc_matrix(np.array([[0], [1]], dtype=np.complex128))

        # Set up coin schedule
        if isinstance(coin_schedule, CoinSchedule):
            self.coin_schedule = coin_schedule
        elif isinstance(coin_schedule, dict):
            self.coin_schedule = CoinSchedule("alternating", coins=coin_schedule)
        elif isinstance(coin_schedule, np.ndarray):
            self.coin_schedule = CoinSchedule("fixed",
                                            coins={"default": coin_schedule})
        else:
            # Default to Hadamard
            self.coin_schedule = CoinSchedule("fixed",
                                            coins={"default": create_hadamard().toarray()})

        # Store custom shift function
        self.custom_shift = custom_shift

    def _validate_quantum_state(self, state: csc_matrix) -> None:
        """
        Verify quantum state properties.

        Args:
            state (csc_matrix): Quantum state to validate

        Raises:
            ValueError: If state is invalid
        """
        state_dense = state.toarray().flatten()

        # Check normalization
        norm = np.sum(np.abs(state_dense)**2)
        if not np.allclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"State not normalized: norm = {norm}")

        # Check state vector size
        expected_size = self.P * 2
        if state_dense.size != expected_size:
            raise ValueError(f"Invalid state size: {state_dense.size}, expected {expected_size}")

    def calculate_probabilities(self, psi: csc_matrix) -> np.ndarray:
        """
        Calculate position probabilities from state vector.

        Args:
            psi (csc_matrix): Quantum state vector

        Returns:
            np.ndarray: Position probability distribution

        Raises:
            ValueError: If probability is not conserved
        """
        psi_dense = psi.toarray().flatten()
        prob = np.zeros(self.P)

        # Sum probabilities for both coin states at each position
        for k in range(self.P):
            coin_states = psi_dense[k*2:(k+1)*2]
            prob[k] = np.abs(coin_states[0])**2 + np.abs(coin_states[1])**2

        # Verify probability conservation
        if not np.allclose(np.sum(prob), 1.0, atol=1e-10):
            raise ValueError("Probability not conserved")

        return prob

    def simulate(self, store_states: bool = False) -> Union[
            Tuple[np.ndarray, float],
            Tuple[np.ndarray, float, List[np.ndarray]],
            Tuple[np.ndarray, float, List[np.ndarray], List[np.ndarray]]]:
        """
        Perform quantum walk simulation with dynamic coins.

        Args:
            store_states (bool): Whether to store intermediate states

        Returns:
            Tuple containing:
                - Final probability distribution
                - Execution time
                - (Optional) List of intermediate states if store_states=True
                - List of probability distributions at each step
        """
        start_time = time.time()

        # Initialize state
        psi = kron(self.posn0, self.initial_coin_state)
        self._validate_quantum_state(psi)

        # Create shift operators
        shift_plus, shift_minus = create_sparse_shift_operators(self.P, self.custom_shift)
        eye_P = csc_matrix(np.eye(self.P))

        # Store states and probabilities if requested
        states = [psi.toarray()] if store_states else None
        prob_history = [self.calculate_probabilities(psi)]

        # Evolve system with dynamic coins
        for step in tqdm(range(self.N), desc="Evolution"):
            # Get coin operator for this step
            coin = self.coin_schedule.get_coin(step)

            # Create evolution operator for this step
            coin_op = kron(eye_P, coin)

            # Create shift operator
            outer_prod0 = self.coin0.dot(self.coin0.T.conjugate())
            outer_prod1 = self.coin1.dot(self.coin1.T.conjugate())
            shift_op = kron(shift_plus, outer_prod0) + kron(shift_minus, outer_prod1)

            # Evolution
            U = shift_op.dot(coin_op)
            psi = U.dot(psi)

            self._validate_quantum_state(psi)
            if store_states:
                states.append(psi.toarray())

            prob_history.append(self.calculate_probabilities(psi))

        # Calculate final probabilities
        prob = prob_history[-1]

        execution_time = time.time() - start_time
        if store_states:
            return prob, execution_time, states, prob_history
        return prob, execution_time, prob_history

    def analyze_statistics(self, prob: np.ndarray) -> dict:
        """
        Analyze statistical properties of the quantum walk.

        Args:
            prob (np.ndarray): Probability distribution

        Returns:
            dict: Statistical measures including:
                - mean_position
                - standard_deviation
                - variance
                - max_probability
                - min_probability
                - entropy
        """
        positions = np.arange(-self.N, self.N + 1)
        mean_position = np.average(positions, weights=prob)
        variance = np.average((positions - mean_position)**2, weights=prob)
        std_dev = np.sqrt(variance)

        return {
            "mean_position": mean_position,
            "standard_deviation": std_dev,
            "variance": variance,
            "max_probability": np.max(prob),
            "min_probability": np.min(prob),
            "entropy": -np.sum(prob * np.log2(prob + 1e-15))
        }

    def plot_results(self, prob: np.ndarray, exec_time: float):
        """
        Plot results of the quantum walk.

        Args:
            prob (np.ndarray): Probability distribution
            exec_time (float): Execution time
        """
        positions = np.arange(-self.N, self.N + 1)
        plot_distribution(positions, prob, self.N, exec_time)
