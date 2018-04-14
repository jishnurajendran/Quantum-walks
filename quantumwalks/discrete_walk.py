import numpy as np
import time
from typing import Tuple, Optional
from tqdm import tqdm
from .utils.operators import create_hadamard, create_shift_operators
from .utils.visualization import plot_distribution

class QuantumWalk:
    def __init__(self,
                 num_steps: int,
                 initial_coin_state: Optional[np.ndarray] = None,
                 coin_operator: Optional[np.ndarray] = None):
        """
        Initialize Quantum Walk parameters

        Args:
            num_steps (int): Number of steps for the quantum walk
            initial_coin_state (np.ndarray, optional): Custom initial coin state.
                Default: (coin0 - i*coin1)/√2
            coin_operator (np.ndarray, optional): Custom 2x2 coin operator matrix.
                Default: Hadamard operator
        """
        self.N = num_steps
        self.P = 2 * num_steps + 1  # Total number of positions

        # Validate and set custom coin operator
        if coin_operator is not None:
            if coin_operator.shape != (2, 2):
                raise ValueError("Coin operator must be a 2x2 matrix")
            self.custom_coin = coin_operator
        else:
            self.custom_coin = None

        # Setup operators first (to initialize coin0 and coin1)
        self.setup_operators()

        # Validate and set custom initial coin state
        if initial_coin_state is not None:
            if initial_coin_state.shape != (2,):
                raise ValueError("Initial coin state must be a 2D vector")
            if not np.isclose(np.linalg.norm(initial_coin_state), 1):
                raise ValueError("Initial coin state must be normalized")
            self.initial_coin_state = initial_coin_state
        else:
            # Default initial state: (coin0 - i*coin1)/√2
            self.initial_coin_state = (self.coin0 - 1j*self.coin1)/np.sqrt(2)

    def setup_operators(self):
        """Setup quantum operators and initial states"""
        # Coin basis states
        self.coin0 = np.array([1, 0], dtype=np.complex128)
        self.coin1 = np.array([0, 1], dtype=np.complex128)

        # Coin operator (Hadamard by default)
        self.H = self.custom_coin if self.custom_coin is not None else create_hadamard()

        # Shift operators
        self.shift_plus, self.shift_minus = create_shift_operators(self.P)

    def evolve_state(self, state: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Evolve quantum state using evolution operator

        Args:
            state (np.ndarray): Current quantum state
            U (np.ndarray): Evolution operator

        Returns:
            np.ndarray: Evolved quantum state
        """
        return U.dot(state)

    def calculate_evolution_operator(self) -> np.ndarray:
        """
        Calculate the evolution operator

        Returns:
            np.ndarray: Complete evolution operator
        """
        # Coin operator
        coin_op = np.kron(np.eye(self.P), self.H)

        # Shift operator
        shift_op = (np.kron(self.shift_plus, np.outer(self.coin0, self.coin0)) +
                   np.kron(self.shift_minus, np.outer(self.coin1, self.coin1)))

        return shift_op.dot(coin_op)

    def simulate(self) -> Tuple[np.ndarray, float]:
        """
        Perform quantum walk simulation

        Returns:
            Tuple[np.ndarray, float]: Probability distribution and execution time
        """
        start_time = time.time()

        # Initial state
        posn0 = np.zeros(self.P)
        posn0[self.N] = 1  # Start at center
        psi0 = np.kron(posn0, self.initial_coin_state)

        # Evolution operator
        U = self.calculate_evolution_operator()

        # Evolve the system
        psiN = np.linalg.matrix_power(U, self.N).dot(psi0)

        # Calculate probabilities
        prob = np.zeros(self.P)
        for k in tqdm(range(self.P), desc="Calculating probabilities"):
            posn = np.zeros(self.P)
            posn[k] = 1
            M_hat_k = np.kron(np.outer(posn, posn), np.eye(2))
            proj = M_hat_k.dot(psiN)
            prob[k] = np.real(proj.dot(np.conjugate(proj)))

        execution_time = time.time() - start_time
        return prob, execution_time

    def analyze_statistics(self, prob: np.ndarray) -> dict:
        """
        Analyze statistical properties of the quantum walk

        Args:
            prob (np.ndarray): Probability distribution

        Returns:
            dict: Dictionary containing statistical measures
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
            "min_probability": np.min(prob)
        }

    def plot_results(self, prob: np.ndarray, exec_time: float):
        """
        Plot the probability distribution

        Args:
            prob (np.ndarray): Probability distribution
            exec_time (float): Execution time
        """
        positions = np.arange(-self.N, self.N + 1)
        plot_distribution(positions, prob, self.N, exec_time)


def run_quantum_walk_simulation(steps: int,
                              initial_coin_state: Optional[np.ndarray] = None,
                              coin_operator: Optional[np.ndarray] = None):
    """
    Run complete quantum walk simulation with analysis

    Args:
        steps (int): Number of steps for the quantum walk
        initial_coin_state (np.ndarray, optional): Custom initial coin state
        coin_operator (np.ndarray, optional): Custom coin operator
    """
    # Initialize and run simulation
    qw = QuantumWalk(steps, initial_coin_state, coin_operator)
    prob, exec_time = qw.simulate()

    # Plot results
    qw.plot_results(prob, exec_time)

    # Analyze and print statistics
    stats = qw.analyze_statistics(prob)
    print("\nQuantum Walk Statistics:")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
