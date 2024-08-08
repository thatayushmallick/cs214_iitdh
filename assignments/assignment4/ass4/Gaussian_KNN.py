import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from collections import Counter

num_samples = 100
class_1_mean = np.array([1.0, 1.0])
class_2_mean = np.array([-2.0, -2.0])

class_1_cov = np.array([[0.8, 0.4], [0.4, 0.8]])
class_2_cov = np.array([[0.8, -0.6], [-0.6, 0.8]])

X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, num_samples)
X_class_2 = np.random.multivariate_normal(class_2_mean, class_2_cov, num_samples)

X_train = np.vstack((X_class_1, X_class_2))
Y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

def knn(X_train, Y_train, p, k=7):
    # Calculate distances vectorized
    distances = np.linalg.norm(X_train - p, axis=1)
    # Get indices of k nearest neighbors
    k_indices = distances.argsort()[:k]
    # Get the labels of the k nearest neighbors
    k_labels = Y_train[k_indices]
    # Count occurrences and return the most common
    return Counter(k_labels).most_common(1)[0][0]

def learn(X_train, Y_train, k, mesh_data):
    return np.array([knn(X_train, Y_train, point, k) for point in mesh_data])

def main():
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = learn(X_train, Y_train, 7, mesh_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K Nearest Neighbours (KNN) Classifier')
    plt.legend(['Class 0', 'Class 1'])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
