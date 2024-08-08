import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data in the form of columns from a csv file. 
rrcsv = pd.read_csv("rrdata.csv")

# The first column contains X, the 2nd column contains bias, the 3rd column contains Y. 
# The bias column contains only 1 in it
X = np.array(rrcsv.iloc[:, 0])
bias = np.array(rrcsv.iloc[:, 1])
y = np.array(rrcsv.iloc[:, 2])

# We stack the bias and X together to get X. 
X = np.column_stack((bias, X))

# You will implement the closed form of L2 (ridge) regression in this function. 
# This function must return a 2 x 1 vector of weights.   

n_data, n_features = X.shape

def ridge_regression_closed_form(X, y, lamb):
	w = np.matmul(np.linalg.inv(np.matmul(X.T, X) + lamb * np.eye(n_features)), np.matmul(X.T, y))
	return w

# You will implement the gradient descent form of L2 (ridge) regression in this function. 
# This function must return a 2 x 1 vector of weights. 

def ridge_regression_gradient_descent(X, y, lamb, learning_rate, num_iterations):
	w = np.zeros(2)
	for _ in range(num_iterations):
		dw = (-2 / n_data) * np.matmul(X.T, y - np.matmul(X, w)) + 2 * lamb * w
		w = w - dw * learning_rate
	return w

lamb = 0.01

ridge_weights = ridge_regression_closed_form(X, y, lamb)
ridge_weights_gd = ridge_regression_gradient_descent(X, y, lamb, 0.1, 1000)


# The following code will be helpful to plot the results. 
figure = plt.figure()
figure.subplots_adjust(wspace=0.5)
ax1 = figure.add_subplot(1, 2, 1)
ax2 = figure.add_subplot(1, 2, 2)

# You will compare your results by plotting the values of y given in the dataset 
# and those obtained from the closed form solution

ax1.scatter(X[:, 1], y, label="Original Data")
ax1.plot(X[:, 1], X.dot(ridge_weights), color='red', label="Closed Form Solution")
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.legend()
ax1.set_title('Closed Form solution')

# You will compare your results by plotting the values of y given in the dataset 
# and those obtained from the gradient descent solution

ax2.scatter(X[:, 1], y, label="Original Data")
ax2.plot(X[:, 1], X.dot(ridge_weights_gd), color='red', label="Gradient Descent Final")
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.legend()
ax2.set_title('Gradient Descent solution')

figure.savefig("ridgeplots.png")

