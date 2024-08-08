import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import Counter

def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    features = df[['Height (in cms)', 'Weight (in kgs)']].values
    labels = (df['T Shirt Size'] == 'L').astype(int).values  # Convert 'M' to 0 and 'L' to 1
    return features, labels

def knn(X_train, Y_train, point, k=7):
    distances = [distance.euclidean(x, point) for x in X_train]
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = Y_train[nearest_indices]
    return Counter(nearest_labels).most_common(1)[0][0]

def predict_grid(X_train, Y_train, k, grid):
    return np.array([knn(X_train, Y_train, point, k) for point in grid])

def plot_decision_boundary(X, y, Z, xx, yy):
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3, levels=np.arange(-1, 2), colors=('blue', 'red'))
    plt.xlabel('Height (in cms)')
    plt.ylabel('Weight (in kgs)')
    plt.title('K-Nearest Neighbors Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(filename='TShirt_size.csv', k=7):
    X_train, Y_train = load_and_prepare_data(filename)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_grid(X_train, Y_train, k, mesh_points)
    plot_decision_boundary(X_train, Y_train, Z, xx, yy)

if __name__ == "__main__":
    main()
