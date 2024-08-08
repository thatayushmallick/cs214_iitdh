import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

def generate_data(k1, k2, num_samples):
    points_class_1 = np.random.chisquare(k1, (num_samples, 2))
    points_class_2 = np.random.chisquare(k2, (num_samples, 2))
    X_train = np.vstack((points_class_1, points_class_2))
    Y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))
    return X_train, Y_train

def calculate_means_and_covariances(X_train, Y_train):
    mean_class_1 = X_train[Y_train == 0].mean(axis=0)
    mean_class_2 = X_train[Y_train == 1].mean(axis=0)
    cov_class_1 = np.cov(X_train[Y_train == 0].T)
    cov_class_2 = np.cov(X_train[Y_train == 1].T)
    inv_cov_class_1 = np.linalg.inv(cov_class_1)
    inv_cov_class_2 = np.linalg.inv(cov_class_2)
    return (mean_class_1, inv_cov_class_1), (mean_class_2, inv_cov_class_2)

def classify_point(mean1, inv_cov1, mean2, inv_cov2, point):
    dist_to_class_1 = mahalanobis(point, mean1, inv_cov1)
    dist_to_class_2 = mahalanobis(point, mean2, inv_cov2)
    return 1 if dist_to_class_1 > dist_to_class_2 else 0

def predict_mesh(X_train, Y_train, mesh_points):
    (mean_class_1, inv_cov_class_1), (mean_class_2, inv_cov_class_2) = calculate_means_and_covariances(X_train, Y_train)
    predictions = [classify_point(mean_class_1, inv_cov_class_1, mean_class_2, inv_cov_class_2, point) for point in mesh_points]
    return np.array(predictions)

def plot_decision_boundary(X_train, Y_train, Z, xx, yy):
    plt.scatter(X_train[Y_train == 0][:, 0], X_train[Y_train == 0][:, 1], label="Class 1", c='blue')
    plt.scatter(X_train[Y_train == 1][:, 0], X_train[Y_train == 1][:, 1], label="Class 2", c='red')
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3, levels=np.unique(Z), colors=['blue', 'red'])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Learning with Mahalanobis Distance Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    k1, k2, num_samples = 7, 10, 100
    X_train, Y_train = generate_data(k1, k2, num_samples)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_predictions = predict_mesh(X_train, Y_train, np.c_[xx.ravel(), yy.ravel()])
    plot_decision_boundary(X_train, Y_train, mesh_predictions, xx, yy)

if __name__ == "__main__":
    main()
