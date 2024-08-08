import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples):
    class_1_mean = np.array([1.0, 1.0])
    class_2_mean = np.array([-2.0, -2.0])
    class_1_cov = np.array([[0.8, 0.4], [0.4, 0.8]])
    class_2_cov = np.array([[0.8, -0.6], [-0.6, 0.8]])

    X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, num_samples)
    X_class_2 = np.random.multivariate_normal(class_2_mean, class_2_cov, num_samples)
    X_train = np.vstack((X_class_1, X_class_2))
    Y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    return X_train, Y_train

def find_means(X_train, Y_train):
    return [X_train[Y_train == i].mean(axis=0) for i in range(2)]

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def classify_point(means, point):
    distances = [euclidean_distance(mean, point) for mean in means]
    return np.argmin(distances)

def calculate_accuracy(X_train, Y_train, means):
    predictions = [classify_point(means, x) for x in X_train]
    accuracy = (predictions == Y_train).mean() * 100
    return accuracy

def learn_and_predict(X_train, Y_train, mesh_data):
    means = find_means(X_train, Y_train)
    accuracy = calculate_accuracy(X_train, Y_train, means)
    print(f"Accuracy: {accuracy}%")

    if mesh_data is not None:
        return np.array([classify_point(means, point) for point in mesh_data])

def plot_decision_boundary(X_train, Y_train, Z, xx, yy):
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3, levels=np.arange(-1, 2), colors=('blue', 'red'))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Learning with Prototype (LwP) Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    num_samples = 100
    X_train, Y_train = generate_data(num_samples)

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = learn_and_predict(X_train, Y_train, np.c_[xx.ravel(), yy.ravel()])
    plot_decision_boundary(X_train, Y_train, Z, xx, yy)

if __name__ == "__main__":
    main()
