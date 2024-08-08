import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    features = df[['Height (in cms)', 'Weight (in kgs)']].values
    labels = (df['T Shirt Size'] == 'L').astype(int).values  # 'M' as 0, 'L' as 1
    return features, labels

def calculate_means(X_train, Y_train):
    mean_vectors = [X_train[Y_train == i].mean(axis=0) for i in np.unique(Y_train)]
    return mean_vectors

def classify_point(means, point):
    distances = [np.linalg.norm(point - mean) for mean in means]
    return np.argmin(distances)

def predict(X_train, Y_train, mesh_data):
    means = calculate_means(X_train, Y_train)
    return np.array([classify_point(means, point) for point in mesh_data])

def plot_decision_boundary(X_train, Y_train, Z, xx, yy):
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)
    plt.xlabel('Height (in cms)')
    plt.ylabel('Weight (in kgs)')
    plt.title('Learning with Prototype (LwP) Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(filename='TShirt_size.csv'):
    X_train, Y_train = load_data(filename)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    mesh_predictions = predict(X_train, Y_train, np.c_[xx.ravel(), yy.ravel()])
    plot_decision_boundary(X_train, Y_train, mesh_predictions, xx, yy)

if __name__ == "__main__":
    main()
