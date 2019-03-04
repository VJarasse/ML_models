import numpy as np
from cluster import *

# Import you data here
data_path = "test.txt"


def k_means(data, k):
    """
    Actual K-mean algorithm
    :param data: the data to run the algorithm on. Should be a numpy array.
    :param k: the number of clusters you want to use.
    :return: The list of cluster centers
    """
    # Initialise the clusters:
    