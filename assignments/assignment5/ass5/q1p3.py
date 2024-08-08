import numpy as np
import matplotlib.pyplot as plt

def generate_noisy_sine_data(num_points, x_range, noise_level):
    """Generate noisy observations from a sine function."""
    X = np.random.uniform(x_range[0], x_range[1], num_points)
    y_true = np.sin(X)
    noise = noise_level * np.random.randn(num_points)
    y_noisy = y_true + noise
    return X, y_noisy, y_true

def polynomial_features(X, degree):
    """Generate polynomial features up to a given degree."""
    return np.column_stack([X ** d for d in range(degree + 1)])

def ridge_regression(X, y, lambda_val):
    """Compute Ridge Regression weights."""
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y

def plot_data_and_fit(X, y_noisy, y_true, X_dense, y_pred):
    """Plot the noisy data, true function, and regression fit."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_noisy, label='Noisy Observations', alpha=0.7, edgecolor='black', color='blue')
    plt.plot(X_dense, y_true, 'r--', label='True Function', linewidth=2.5)
    plt.plot(X_dense, y_pred, 'g', label='Ridge Regression Fit', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ridge Regression with Polynomial Features')
    plt.legend()
    plt.grid(True)
    plt.show()

# Set the random seed for reproducibility
np.random.seed(0)

# Generate synthetic data based on a noisy sine function
num_samples = 80
x_range = (0, 5)
noise_level = 0.5
X, y_noisy, _ = generate_noisy_sine_data(num_samples, x_range, noise_level)

# Transform features into polynomial features
polynomial_degree = 5
X_poly = polynomial_features(X, polynomial_degree)

# Perform Ridge Regression
lambda_val = 0.1
ridge_weights = ridge_regression(X_poly, y_noisy, lambda_val)

# Prepare dense x values for smooth plot lines
X_dense = np.linspace(x_range[0], x_range[1], 400)
X_dense_poly = polynomial_features(X_dense, polynomial_degree)
y_dense_true = np.sin(X_dense)

# Predict using the Ridge Regression model
y_dense_pred = X_dense_poly @ ridge_weights

# Plot the results
plot_data_and_fit(X, y_noisy, y_dense_true, X_dense, y_dense_pred)