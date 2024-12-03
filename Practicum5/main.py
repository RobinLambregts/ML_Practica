import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Number of Clusters for K-Means
k = 2
clusterList = [[] for _ in range(k)]

# Generate data
data, labels = make_moons(100, noise=0.1)  # 100 samples with some noise


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_neighbors(data, point_idx, eps):
    """Find all points within eps distance from the point."""
    neighbors = []
    for i, point in enumerate(data):
        if calculate_distance(data[point_idx], point) < eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_samples):
    """Expand the cluster by adding density-reachable points."""
    labels[point_idx] = cluster_id
    to_visit = neighbors.copy()
    to_visit.remove(point_idx)  # Remove the point itself from the list of neighbors

    while to_visit:
        current_point = to_visit.pop()
        if labels[current_point] == -1:
            labels[current_point] = cluster_id  # Mark as noise and then move it to this cluster
        if labels[current_point] == 0:  # Unvisited point
            labels[current_point] = cluster_id
            current_neighbors = get_neighbors(data, current_point, eps)
            if len(current_neighbors) >= min_samples:
                to_visit.extend(current_neighbors)


def calculate_dbscan(data, eps=0.2, min_samples=5):
    """Perform DBSCAN clustering without using external libraries."""
    labels = np.zeros(len(data))  # 0 means unvisited, -1 means noise
    cluster_id = 0

    for point_idx in range(len(data)):
        if labels[point_idx] == 0:  # If the point is unvisited
            neighbors = get_neighbors(data, point_idx, eps)
            if len(neighbors) < min_samples:
                labels[point_idx] = -1  # Mark as noise
            else:
                cluster_id += 1  # Found a new cluster
                expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_samples)

    return labels


def distance(data, centroids):
    """Assign points to the nearest centroid."""
    clusterList = [[] for _ in range(k)]
    for datapoint in data:
        min_dist = float('inf')
        index = -1
        for i, centroid in enumerate(centroids):
            dist = calculate_distance(centroid, datapoint)
            if dist < min_dist:
                min_dist = dist
                index = i
        clusterList[index].append(datapoint)
    return clusterList


def mean(cluster):
    """Calculate the mean of a cluster."""
    if len(cluster) == 0:
        return (0, 0)  # Handle empty cluster
    sum_x = sum(point[0] for point in cluster)
    sum_y = sum(point[1] for point in cluster)
    return sum_x / len(cluster), sum_y / len(cluster)


def calculate_kmeans():
    """Perform K-Means clustering."""
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_indices]  # Initial centroids

    for _ in range(100):  # Maximum 100 iterations
        clusters = distance(data, centroids)  # Assign points to clusters
        new_centroids = []
        for cluster in clusters:
            new_centroids.append(mean(cluster))
        new_centroids = np.array(new_centroids)  # Update centroids

        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return clusters, centroids


def plot_clusters(data, clusters, centroids):
    """Plot the K-Means clustering results."""
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % len(colors)], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_dbscan(data, labels):
    """Plot the DBSCAN clustering results."""
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Label -1 is considered noise in DBSCAN
            color = 'black'
        mask = labels == label
        plt.scatter(data[mask, 0], data[mask, 1], c=[color], label=f'Cluster {label}', s=100, edgecolor='k')

    plt.title('DBSCAN Clustering (Custom Implementation)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Perform K-Means and plot
    clusters, centroids = calculate_kmeans()
    plot_clusters(data, clusters, centroids)

    # Perform DBSCAN and plot
    dbscan_labels = calculate_dbscan(data, eps=0.2, min_samples=5)
    plot_dbscan(data, dbscan_labels)