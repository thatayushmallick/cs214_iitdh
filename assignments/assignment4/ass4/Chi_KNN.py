import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import Counter

def generate_data(k1, k2, num_samples):
    points_class_1 = np.random.chisquare(k1, (num_samples, 2))
    points_class_2 = np.random.chisquare(k2, (num_samples, 2))
    X_train = np.vstack((points_class_1, points_class_2))
    Y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))
    return X_train, Y_train

def knn(X_train, Y_train, point, k=7):
    distances = cdist(X_train, [point], 'euclidean').flatten()
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = Y_train[nearest_indices]
    return Counter(nearest_labels).most_common(1)[0][0]

def predict_mesh(X_train, Y_train, k, xx, yy):
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.array([knn(X_train, Y_train, point, k) for point in mesh_points])
    return predictions.reshape(xx.shape)

def plot_decision_boundary(X_train, Y_train, Z, xx, yy):
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=('blue', 'red'))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Nearest Neighbors Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(k1=7, k2=10, num_samples=100, k=5):
    X_train, Y_train = generate_data(k1, k2, num_samples)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = predict_mesh(X_train, Y_train, k, xx, yy)
    plot_decision_boundary(X_train, Y_train, Z, xx, yy)

if __name__ == "__main__":
    main()
