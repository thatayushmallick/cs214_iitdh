import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples=50, true_slope=2.5, noise_level=0.5):
    """Generate linear data with added noise."""
    X = np.linspace(0, 10, num_samples)
    noise = noise_level * np.random.randn(num_samples)
    y = X * true_slope + 1 + noise  # True intercept is 1
    return X, y

def add_bias_term(X):
    """Add a bias term (column of ones) to the dataset."""
    return np.c_[np.ones(X.shape[0]), X]

def ridge_regression_closed_form(X, y, lambda_val=0.01):
    """Compute Ridge Regression weights using the closed-form solution."""
    X_with_bias = add_bias_term(X)
    regularization_term = lambda_val * np.eye(X_with_bias.shape[1])
    w = np.linalg.inv(X_with_bias.T @ X_with_bias + regularization_term) @ X_with_bias.T @ y
    return w

def ridge_regression_gradient_descent(X, y, alpha=0.001, epochs=1000, lambda_val=0.01):
    """Compute Ridge Regression weights using gradient descent."""
    X_with_bias = add_bias_term(X)
    w = np.random.randn(X_with_bias.shape[1])
    for _ in range(epochs):
        gradient = X_with_bias.T @ (X_with_bias @ w - y) + lambda_val * w
        w -= alpha * gradient
    return w

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
X, y = generate_data()

# Compute weights using closed-form solution
weights_closed_form = ridge_regression_closed_form(X, y)

# Compute weights using gradient descent
weights_gradient_descent = ridge_regression_gradient_descent(X, y)

# Plot results
# plot_regression(X, y, weights_closed_form, 'Ridge Regression (Closed-form Solution)')
# plot_regression(X, y, weights_gradient_descent, 'Ridge Regression (Gradient Descent Solution)', initial_weights=np.zeros(X.shape[1] + 1))

plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='Data')
plt.plot(X, add_bias_term(X) @ weights_closed_form, color='red', label='Ridge Regression (Closed-form Solution)')
plt.title('Ridge Regression (Closed-form Solution)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='Data')
X_with_bias = add_bias_term(X)  # Add bias term to X for plotting
# Ensure initial weights are of correct dimension (2,) for the bias and the slope
initial_weights = np.zeros(X_with_bias.shape[1])
plt.plot(X, X_with_bias @ initial_weights, color='green', label='Initial Line')
plt.plot(X, X_with_bias @ weights_gradient_descent, color='red', label='Ridge Regression (Gradient Descent Solution)')
plt.title('Ridge Regression (Gradient Descent Solution)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()