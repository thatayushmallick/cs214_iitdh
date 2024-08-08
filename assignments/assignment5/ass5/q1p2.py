import numpy as np
import matplotlib.pyplot as plt

def set_random_seed(seed=0):
    """Set the seed for NumPy's random number generator for reproducibility."""
    np.random.seed(seed)

def generate_noisy_sine_data(num_samples=80, scale_factor=5, noise_variance=0.5):
    """Generate noisy observations of a sine function.

    Args:
        num_samples (int): Number of data points to generate.
        scale_factor (float): Factor to scale the X values to widen their range.
        noise_variance (float): Variance of the Gaussian noise added to the sine values.

    Returns:
        tuple: Tuple containing:
            - X (numpy.ndarray): Array of X values.
            - y_noisy (numpy.ndarray): Array of noisy y values corresponding to the sine of X.
            - y_true (numpy.ndarray): Array of true y values without noise.
    """
    # Generate random X values within a uniform distribution, then scale
    X = np.random.uniform(0, 1, num_samples) * scale_factor
    # Calculate true y values based on the sine function
    y_true = np.sin(X)
    # Add Gaussian noise to y values to simulate real-world observations
    noise = noise_variance * np.random.randn(num_samples)
    y_noisy = y_true + noise
    return X, y_noisy, y_true

def plot_noisy_sine_data(X, y_noisy, y_true):
    """Plot the noisy sine data and the true sine function.

    Args:
        X (numpy.ndarray): Array of X values.
        y_noisy (numpy.ndarray): Array of noisy y values.
        y_true (numpy.ndarray): Array of true y values without noise.
    """
    plt.figure(figsize=(10, 6))
    # Plot noisy observations
    plt.scatter(X, y_noisy, label='Noisy Observations', alpha=0.6, edgecolor='black')
    # Plot the true sine function for comparison
    plt.plot(np.sort(X), np.sin(np.sort(X)), 'r', label='True Sine Function', linewidth=2)
    plt.xlabel('X Values')
    plt.ylabel('y Values')
    plt.title('Sine Function with Superimposed Noisy Data')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main script execution
if __name__ == "__main__":
    set_random_seed()
    X, y_noisy, y_true = generate_noisy_sine_data()
    plot_noisy_sine_data(X, y_noisy, y_true)