import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science','notebook'])

def plot_distribution(positions: np.ndarray, prob: np.ndarray,
                     steps: int, exec_time: float):
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
    plt.title(f'Quantum Walk Distribution after {steps} steps', fontsize=14)

    # Add statistical information
    plt.text(0.02, 0.98, f'Median Position: {np.median(positions):.4f}\n'
                        f'Mean Position: {np.average(positions, weights=prob):.4f}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')

    plt.legend()
    plt.tight_layout()
    plt.show()
