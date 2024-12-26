# Example usage with custom components
def custom_shift_function(size):
    """Custom shift operator that implements periodic boundary conditions"""
    shift_plus = np.roll(np.eye(size), 1, axis=0)
    shift_minus = np.roll(np.eye(size), -1, axis=0)
    return shift_plus, shift_minus

# Custom coin operator (Grover coin)
grover_coin = np.array([
    [2/3, 1/3],
    [1/3, -2/3]
]) * np.sqrt(2)

# Custom initial states
custom_position = np.zeros(2001)  # for 1000 steps
custom_position[1000] = 1  # Start at center
custom_coin = np.array([1, 1]) / np.sqrt(2)  # Equal superposition

# Create quantum walk with custom components
qw = QuantumWalk(
    num_steps=1000,
    initial_position=custom_position,
    initial_coin_state=custom_coin,
    coin_operator=grover_coin,
    custom_shift=custom_shift_function
)

# Run simulation
prob, exec_time = qw.simulate()
qw.plot_results(prob, exec_time)
