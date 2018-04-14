# %%
import numpy as np
import matplotlib.pyplot as plt
from quantumwalks import run_quantum_walk_simulation
from quantumwalks.discrete_walk import QuantumWalk
from quantumwalks.utils.operators import create_shift_operators

def create_custom_coin(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Create a custom coin operator U(α,β,γ) using vectorized operations.
    """
    return np.array([
        [np.exp(1j * alpha) * np.cos(beta), -np.exp(-1j * gamma) * np.sin(beta)],
        [np.exp(1j * gamma) * np.sin(beta), np.exp(-1j * alpha) * np.cos(beta)]
    ], dtype=np.complex128)

def calculate_lr_difference_alternating_coins(qw: QuantumWalk, coin_A: np.ndarray, coin_B: np.ndarray) -> np.ndarray:
    """
    Calculate P_L - P_R for each step of the walk using alternating coins A and B.
    """
    # Initialize state
    posn0 = np.zeros(qw.P)
    posn0[qw.N] = 1
    psi = np.kron(posn0, qw.initial_coin_state)

    # Pre-allocate arrays
    differences = np.zeros(qw.N + 1)
    positions = np.arange(-qw.N, qw.N + 1)

    # Create measurement operators matrix once
    measurement_ops = np.array([
        np.kron(np.outer(np.eye(qw.P)[i], np.eye(qw.P)[i]), np.eye(2))
        for i in range(qw.P)
    ])

    # Create shift operators using the utility function
    shift_plus, shift_minus = create_shift_operators(qw.P)
    shift_op = (np.kron(shift_plus, np.outer(qw.coin0, qw.coin0)) +
                np.kron(shift_minus, np.outer(qw.coin1, qw.coin1)))

    # Calculate evolution operators for both coins
    U_A = shift_op @ np.kron(np.eye(qw.P), coin_A)
    U_B = shift_op @ np.kron(np.eye(qw.P), coin_B)

    # Calculate for each step
    for step in range(qw.N + 1):
        # Vectorized probability calculation
        projections = measurement_ops @ psi
        prob = np.real(np.sum(projections * np.conjugate(projections), axis=1))

        # Vectorized P_L and P_R calculation
        P_L = np.sum(prob[positions < 0])
        P_R = np.sum(prob[positions >= 0])
        differences[step] = P_L - P_R

        # Evolve state with alternating coin operators
        if step < qw.N:
            if step % 2 == 0:
                psi = U_A @ psi  # Apply coin A
            else:
                psi = U_B @ psi  # Apply coin B

    return differences, prob  # Return both differences and final probability distribution

def run_simulation_alternating_coins(steps: int, 
                                   alpha_A: float, beta_A: float, gamma_A: float,
                                   alpha_B: float, beta_B: float, gamma_B: float):
    """
    Run simulation with alternating coins A and B.
    """
    # Create both coin operators
    coin_A = create_custom_coin(alpha_A, beta_A, gamma_A)
    coin_B = create_custom_coin(alpha_B, beta_B, gamma_B)
    initial_state = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)

    print(f"Running quantum walk simulation with alternating coins:")
    print(f"Coin A parameters: α={alpha_A:.2f}, β={beta_A:.2f}, γ={gamma_A:.2f}")
    print(f"Coin B parameters: α={alpha_B:.2f}, β={beta_B:.2f}, γ={gamma_B:.2f}")

    # Create quantum walk object (using coin_A as default)
    qw = QuantumWalk(steps, initial_state, coin_A)

    # Calculate differences and probabilities using alternating coins
    differences, prob = calculate_lr_difference_alternating_coins(qw, coin_A, coin_B)

    # Plot results
    plot_results(qw, prob, 0.0, differences, steps)  # Using 0.0 as execution time placeholder

    return qw, differences, prob

def plot_results(qw: QuantumWalk, prob: np.ndarray,
                exec_time: float, differences: np.ndarray, steps: int):
    """
    Plot both probability distribution and P_L - P_R differences.
    """
    # Plot probability distribution
    positions = np.arange(-qw.N, qw.N + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(positions, prob, 'b-', label='Probability', linewidth=1.5)
    plt.plot(positions, prob, 'ro', markersize=4)
    plt.fill_between(positions, prob, alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Quantum Walk Distribution after {steps} steps\n(Alternating Coins)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot P_L - P_R differences
    plt.figure(figsize=(12, 8))
    x = np.arange(steps + 1)
    plt.plot(x, differences, 'b-', label='P_L - P_R', linewidth=1.5)
    plt.plot(x, differences, 'ro', markersize=4)

    # Set y-axis range from -1 to 1
    plt.ylim(-1, 1)

    plt.grid(True, alpha=0.3)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('P_L - P_R', fontsize=12)
    plt.title('Difference between Left and Right Probabilities\n(Alternating Coins)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%

if __name__ == "__main__":
    # Set parameters
    steps = 800
    
    # Parameters for Coin A
    alpha_A = -51.0/180 * np.pi  # Converting to radians
    beta_A = 45.0/180 * np.pi
    gamma_A = 0.0/180 * np.pi
    
    # Parameters for Coin B
    alpha_B = 0.0/180 * np.pi
    beta_B = 88.0/180 * np.pi
    gamma_B = -16.0/180 * np.pi

    # Run simulation with alternating coins
    qw, differences, prob = run_simulation_alternating_coins(
        steps, 
        alpha_A, beta_A, gamma_A,
        alpha_B, beta_B, gamma_B
    )

    # Print some statistics
    positions = np.arange(-qw.N, qw.N + 1)
    mean_pos = np.average(positions, weights=prob)
    std_dev = np.sqrt(np.average((positions - mean_pos)**2, weights=prob))
    
    print("\nFinal Statistics:")
    print(f"Mean Position: {mean_pos:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Max Probability: {np.max(prob):.4f}")

# %%



