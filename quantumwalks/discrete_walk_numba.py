import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from numba import jit
import time
from tqdm import tqdm

class QuantumWalk:
    def __init__(self, num_steps: int):
        """
        Initialize Quantum Walk parameters
        """
        self.N = num_steps
        self.P = 2 * num_steps + 1
        self.setup_operators()

    @staticmethod
    @jit(nopython=True)
    def create_hadamard() -> np.ndarray:
        """Create Hadamard operator with Numba optimization"""
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def setup_operators(self):
        """Setup quantum operators"""
        # Coin basis states
        self.coin0 = np.array([1, 0], dtype=np.complex128)
        self.coin1 = np.array([0, 1], dtype=np.complex128)

        # Hadamard operator
        self.H = self.create_hadamard()
        
        # Shift operators
        self.shift_plus = np.roll(np.eye(self.P), 1, axis=0)
        self.shift_minus = np.roll(np.eye(self.P), -1, axis=0)

    @jit(nopython=True)
    def evolve_state(self, state: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Evolve quantum state using evolution operator"""
        return U.dot(state)

    def calculate_evolution_operator(self) -> np.ndarray:
        """Calculate the evolution operator"""
        # Coin operator
        coin_op = np.kron(np.eye(self.P), self.H)
        
        # Shift operator
        shift_op = (np.kron(self.shift_plus, np.outer(self.coin0, self.coin0)) + 
                   np.kron(self.shift_minus, np.outer(self.coin1, self.coin1)))
        
        return shift_op.dot(coin_op)

    def simulate(self) -> Tuple[np.ndarray, float]:
        """
        Perform quantum walk simulation
        Returns: probability distribution and execution time
        """
        start_time = time.time()
        
        # Initial state
        posn0 = np.zeros(self.P)
        posn0[self.N] = 1  # Start at center
        psi0 = np.kron(posn0, (self.coin0 + self.coin1)/np.sqrt(2))

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

    def plot_distribution(self, prob: np.ndarray, exec_time: float):
        """
        Plot the probability distribution with enhanced visualization
        """
        positions = np.arange(-self.N, self.N + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Main probability distribution
        plt.plot(positions, prob, 'b-', label='Probability', linewidth=1.5)
        plt.plot(positions, prob, 'ro', markersize=4)
        
        # Fill area under the curve
        plt.fill_between(positions, prob, alpha=0.3)
        
        # Formatting
        plt.grid(True, alpha=0.3)
        plt.xlabel('Position', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Quantum Walk Distribution after {self.N} steps\n'
                 f'Execution Time: {exec_time:.2f} seconds', fontsize=14)
        
        # Add statistical information
        plt.text(0.02, 0.98, f'Standard Deviation: {np.std(prob):.4f}\n'
                            f'Mean Position: {np.average(positions, weights=prob):.4f}',
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

    def analyze_statistics(self, prob: np.ndarray) -> dict:
        """
        Analyze statistical properties of the quantum walk
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

def run_quantum_walk_simulation(steps: int):
    """
    Run complete quantum walk simulation with analysis
    """
    # Initialize and run simulation
    qw = QuantumWalk(steps)
    prob, exec_time = qw.simulate()
    
    # Plot results
    qw.plot_distribution(prob, exec_time)
    
    # Analyze and print statistics
    stats = qw.analyze_statistics(prob)
    print("\nQuantum Walk Statistics:")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
