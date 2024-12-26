import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_distribution(positions: np.ndarray, prob: np.ndarray,
                     steps: int, exec_time: float):
    """Plot quantum walk distribution"""
    try:
        import scienceplots
        plt.style.use(['science', 'notebook'])
    except ImportError:
        print("Warning: scienceplots not installed. Using default style.")

    plt.figure(figsize=(12, 8))
    plt.plot(positions, prob, 'b-', label='Probability', linewidth=1.5)
    plt.plot(positions, prob, 'ro', markersize=4)
    plt.fill_between(positions, prob, alpha=0.3)

    plt.grid(True, alpha=0.3)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Quantum Walk Distribution after {steps} steps\n'
             f'(Execution time: {exec_time:.2f}s)', fontsize=14)

    mean_pos = np.average(positions, weights=prob)
    median_pos = np.median(positions)

    plt.text(0.02, 0.98,
            f'Median Position: {median_pos:.4f}\n'
            f'Mean Position: {mean_pos:.4f}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')

    plt.legend()
    plt.tight_layout()
    plt.show()
