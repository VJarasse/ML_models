import numpy as np
import sys
from cluster import *

# Import you data here
data_path = "test.txt"


def k_means(data, k, iterations):
    """
    Actual K-mean algorithm
    :param data: the data to run the algorithm on. Should be a numpy array.
    :param k: the number of clusters you want to use.
    :return: The list of cluster centers
    """

    # Get min and max of data
    minima, maxima = np.min(data, axis=0), np.max(data, axis=0)

    # Draw cluster centers from uniform distribution
    centers_x = np.random.uniform(minima[0], maxima[0], k)
    centers_y = np.random.uniform(minima[1], maxima[1], k)

    # Initialise the clusters:
    coordinates = [np.array([centers_x[i], centers_y[i]]) for i in range(k)]
    clusters = []
    for center in coordinates:
        clusters.append(cluster(center))
        print(center)

    # Keep track of cluster index for each point
    point_to_cluster = [-1] * data.shape[0]

    # Main loop
    for it in range(iterations):

        # Assign each point to its closest cluster
        for pt_idx in range(data.shape[0]):
            point = data[pt_idx, :]
            closest = get_closest(point, clusters)
            if closest != -1:
                clusters[closest].add_member(pt_idx, point)

                # Remove the point from its previous cluster
                old_idx = point_to_cluster[pt_idx]
                if old_idx != -1:
                    clusters[old_idx].remove_member(old_idx)

                # Update the record of the point cluster
                point_to_cluster[pt_idx] = closest

        # Now that each point has been assigned to its nearest cluster
        # we can update the clusters centers

        for cluster in clusters:
            cluster.update_center()

    for cluster in clusters:
        print(cluster.center)

    return


def get_closest(point, clusters):
    """
    Sub function of K-means algorithm: get closest cluster to a point
    :param point: np array of point coordinates
    :param clusters: list of all cluster instances
    :return: index of closest cluster to the point
    """
    min_dist = sys.maxsize
    min_idx = -1
    for i, cluster in enumerate(clusters):
        dist = np.linalg.norm(point - cluster.center)
        if dist < min_dist:
            min_idx = i
            min_dist = dist
    return min_idx

