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

        # Store states and probabilities if needed
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
            "entropy": 0 - np.sum(prob * np.log2(prob + 1e-15))
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

class TwoQubitQuantumWalk:
    """
    Quantum Walk implementation with two coin qubits.
    The total coin space is 4-dimensional (|00⟩, |01⟩, |10⟩, |11⟩).
    """

    def __init__(self,
                 num_steps: int,
                 initial_position: Optional[Union[int, np.ndarray]] = None,
                 initial_coin_state: Optional[np.ndarray] = None,
                 coin1: Optional[np.ndarray] = None,
                 coin2: Optional[np.ndarray] = None):
        """
        Initialize Two-Qubit Quantum Walk.

        Args:
            num_steps (int): Number of steps for the walk
            initial_position (Optional[Union[int, np.ndarray]]): Initial position
            initial_coin_state (Optional[np.ndarray]): Initial 4D coin state
            coin1 (Optional[np.ndarray]): First coin operator
            coin2 (Optional[np.ndarray]): Second coin operator
        """
        if num_steps < 1:
            raise ValueError("Number of steps must be positive")

        self.N = num_steps
        self.P = 2 * num_steps + 1

        # Create initial position state
        self.posn0, _ = create_initial_state(self.P, initial_position)

        # Create or validate initial coin state (4D)
        if initial_coin_state is None:
            # Default to (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
            self.initial_coin_state = csc_matrix(
                np.ones(4, dtype=np.complex128).reshape(-1, 1) / 2
            )
        else:
            if len(initial_coin_state) != 4:
                raise ValueError("Initial coin state must be 4-dimensional")
            if not np.allclose(np.linalg.norm(initial_coin_state), 1):
                raise ValueError("Initial coin state must be normalized")
            self.initial_coin_state = csc_matrix(
                np.asarray(initial_coin_state, dtype=np.complex128).reshape(-1, 1)
            )

        # Create basis states for the 4D coin space
        self.basis_states = [
            csc_matrix(state.reshape(-1, 1))
            for state in np.eye(4, dtype=np.complex128)
        ]

        # Set coin operators
        self.coin1 = coin1 if coin1 is not None else create_hadamard().toarray()
        self.coin2 = coin2 if coin2 is not None else create_hadamard().toarray()

    def _validate_quantum_state(self, state: csc_matrix) -> None:
        """
        Verify quantum state properties.

        Args:
            state (csc_matrix): State to validate

        Raises:
            ValueError: If state is invalid
        """
        state_dense = state.toarray().flatten()

        # Check normalization
        norm = np.sum(np.abs(state_dense)**2)
        if not np.allclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"State not normalized: norm = {norm}")

        # Check state vector size (4 coin states per position)
        expected_size = self.P * 4
        if state_dense.size != expected_size:
            raise ValueError(f"Invalid state size: {state_dense.size}, expected {expected_size}")

    def create_two_qubit_coin(self, coin1: np.ndarray, coin2: np.ndarray) -> csc_matrix:
        """
        Create combined two-qubit coin operator from individual coin operators.

        Args:
            coin1 (np.ndarray): First coin operator (2x2)
            coin2 (np.ndarray): Second coin operator (2x2)

        Returns:
            csc_matrix: Combined 4x4 coin operator
        """
        return kron(csc_matrix(coin1), csc_matrix(coin2))

    def create_four_way_shift(self) -> csc_matrix:
        """
        Create shift operator for four-way movement based on coin state.

        Returns:
            csc_matrix: Shift operator
        """
        # Create basic shift operators
        shift_right = np.roll(np.eye(self.P), 1, axis=0)
        shift_left = np.roll(np.eye(self.P), -1, axis=0)
        stay = np.eye(self.P)

        # Create projectors onto coin basis states
        P00 = self.basis_states[0].dot(self.basis_states[0].T.conjugate())
        P01 = self.basis_states[1].dot(self.basis_states[1].T.conjugate())
        P10 = self.basis_states[2].dot(self.basis_states[2].T.conjugate())
        P11 = self.basis_states[3].dot(self.basis_states[3].T.conjugate())

        # Combine shift operators with corresponding coin projectors
        shift_op = (kron(csc_matrix(shift_right), P00) +  # |00⟩ → right
                   kron(csc_matrix(stay), P01) +          # |01⟩ → stay
                   kron(csc_matrix(stay), P10) +          # |10⟩ → stay
                   kron(csc_matrix(shift_left), P11))     # |11⟩ → left

        return shift_op

    def simulate(self, num_steps: Optional[int] = None) -> Tuple[np.ndarray, float, list]:
        """Perform quantum walk simulation with two coin qubits."""
        start_time = time.time()

        if num_steps is not None:
            self.N = num_steps

        # Initialize state
        psi = kron(self.posn0, self.initial_coin_state)
        self._validate_quantum_state(psi)

        # Create combined coin operator for the entire system
        coin_op = kron(csc_matrix(np.eye(self.P)),
                       self.create_two_qubit_coin(self.coin1, self.coin2))

        # Create shift operator
        shift_op = self.create_four_way_shift()

        # Store probability history
        prob_history = [self.calculate_probabilities(psi)]

        # Evolution
        for _ in tqdm(range(self.N), desc="Evolution"):
            # Apply coin
            psi = coin_op.dot(psi)

            # Apply shift
            psi = shift_op.dot(psi)

            self._validate_quantum_state(psi)
            prob_history.append(self.calculate_probabilities(psi))

        execution_time = time.time() - start_time
        return self.calculate_probabilities(psi), execution_time, prob_history

    def calculate_probabilities(self, psi: csc_matrix) -> np.ndarray:
        """
        Calculate position probabilities from state vector.

        Args:
            psi (csc_matrix): Quantum state vector

        Returns:
            np.ndarray: Position probability distribution
        """
        psi_dense = psi.toarray().flatten()
        prob = np.zeros(self.P)

        # Sum probabilities for all four coin states at each position
        for k in range(self.P):
            coin_states = psi_dense[k*4:(k+1)*4]
            prob[k] = np.sum(np.abs(coin_states)**2)

        return prob

    def analyze_statistics(self, prob: np.ndarray) -> dict:
        """
        Analyze statistical properties of the two-qubit quantum walk.

        Args:
            prob (np.ndarray): Probability distribution

        Returns:
            dict: Statistical measures
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
            "entropy": 0 - np.sum(prob * np.log2(prob + 1e-15))
        }

    def plot_results(self, prob: np.ndarray, exec_time: float):
        """
        Plot results of the two-qubit quantum walk.

        Args:
            prob (np.ndarray): Probability distribution
            exec_time (float): Execution time
        """
        positions = np.arange(-self.N, self.N + 1)
        plot_distribution(positions, prob, self.N, exec_time)
