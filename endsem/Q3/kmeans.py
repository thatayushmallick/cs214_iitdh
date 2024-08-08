# implement kmeans algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## read the data.csv file provided in the folder
data = pd.read_csv("data.csv")

data = np.array(data.iloc[:,:])

# Define function to calculate euclidean distance between two points
def euclidean(a, b):
	return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def k_means(X, K, max_iters=100):

    # Randomly initialize K centroids at random locations within the data range
    min_vals = np.min(X, axis=0)# find minimum value of input feature X
    max_vals = np.max(X, axis=0)# find maximum value of input feature X
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(K, X.shape[1]))

    # Total number of data points
    n_data = X.shape[0]

    # Store initial centroids for plotting
    initial_centroids = centroids.copy()

    for _ in range(max_iters):
    
        # Calculate the distance between each data point and each centroid
        
        # Initialise matrix containing distances of all data points from all centroids
        distances = np.zeros((n_data, K))

	# Calculate all euclidean distances and store them in matrix
        for i in range(n_data):
            for j in range(len(distances[i])):
                distances[i][j] = euclidean(centroids[j], X[i])

        # Assign each data point to the closest centroid
        labels = np.argmin(distances, axis = 1)

        # Update the centroids as the mean of all data points assigned to them
        new_centroids = np.zeros((K, X.shape[1]))
        cluster_count = np.zeros(K)
        for i in range(n_data):
       	    new_centroids[labels[i]]+=X[i]
       	    cluster_count[labels[i]]+=1
       	for k in range(K):
       	    new_centroids[k] = new_centroids[k] / cluster_count[k]

        # Check for convergence(centroids do not change)
        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return initial_centroids, centroids, labels

# Perform K-means clustering
K = 3
initial_centroids, final_centroids, labels = k_means(data, K)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='blue', marker='X', s=200, label='Initial Centroids')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=200, label='Final Centroids')
plt.legend()
plt.title('K-means Clustering')
plt.savefig('final_clusters.jpg', format='jpg')
plt.show()
