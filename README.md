# Identifying-groups-of-similar-wines
Assignment: identifying groups of similar wines


import numpy as np
import pandas as pd

class Matrix:
    """
    A class to represent a 2D matrix for performing clustering operations.

    Attributes:
        array_2d (np.ndarray): A 2D NumPy array containing the matrix data.
    """

    def __init__(self, filename=None):
        """
        Initializes the Matrix instance.

        Args:
            filename (str, optional): The name of the CSV file to load data from. Defaults to None.
        """
        self.array_2d = None
        if filename:
            self.load_from_csv(filename)

    def load_from_csv(self, filename):
        """
        Loads data from a CSV file into the matrix.

        Args:
            filename (str): The name of the CSV file to read data from.
        """
        # Read CSV file using pandas and convert to numpy array
        data = pd.read_csv(filename)
        self.array_2d = data.values

    def standardize(self):
        """
        Standardizes each column of the matrix to have values between 0 and 1.
        """
        # Standardize each column (axis=0 for columns)
        self.array_2d = (self.array_2d - self.array_2d.mean(axis=0)) / (self.array_2d.max(axis=0) - self.array_2d.min(axis=0))

    def get_distance(self, other_matrix, row_i):
        """
        Calculates the Euclidean distance between a specified row of this matrix and each row in another matrix.

        Args:
            other_matrix (Matrix): The other Matrix instance to calculate distances to.
            row_i (int): The index of the row in this matrix to compare against.

        Returns:
            Matrix: A new Matrix containing the distances as a single column.
        """
        distances = np.linalg.norm(self.array_2d[row_i] - other_matrix.array_2d, axis=1)
        return Matrix.from_array(distances.reshape(-1, 1))

    def get_weighted_distance(self, other_matrix, weights, row_i):
        """
        Calculates the weighted Euclidean distance between a specified row of this matrix and each row in another matrix.

        Args:
            other_matrix (Matrix): The other Matrix instance to calculate distances to.
            weights (Matrix): A Matrix containing the weights for each feature.
            row_i (int): The index of the row in this matrix to compare against.

        Returns:
            Matrix: A new Matrix containing the weighted distances as a single column.
        """
        weights_array = weights.array_2d.flatten()
        weighted_diff = (self.array_2d[row_i] - other_matrix.array_2d) ** 2
        weighted_distances = np.sum(weighted_diff * weights_array, axis=1)
        return Matrix.from_array(weighted_distances.reshape(-1, 1))

    def get_count_frequency(self):
        """
        Counts the frequency of unique values in the matrix.

        Returns:
            dict: A dictionary mapping unique values to their counts, or 0 if the matrix does not have a single column.
        """
        if self.array_2d.shape[1] != 1:
            return 0
        unique, counts = np.unique(self.array_2d, return_counts=True)
        return dict(zip(unique, counts))

    @staticmethod
    def from_array(array):
        """
        Creates a Matrix instance from a given NumPy array.

        Args:
            array (np.ndarray): The array to convert into a Matrix.

        Returns:
            Matrix: A new Matrix instance containing the provided array.
        """
        new_matrix = Matrix()
        new_matrix.array_2d = array
        return new_matrix

    @staticmethod
    def get_initial_weights(m):
        """
        Generates initial weights for the clustering algorithm.

        Args:
            m (int): The number of features.

        Returns:
            Matrix: A new Matrix containing the initial weights.
        """
        weights = np.random.rand(1, m)
        return Matrix.from_array(weights / np.sum(weights))

    @staticmethod
    def get_centroids(data_matrix, K):
        """
        Randomly selects K initial centroids from the data matrix.

        Args:
            data_matrix (Matrix): The data matrix from which to select centroids.
            K (int): The number of centroids to select.

        Returns:
            Matrix: A new Matrix containing the selected centroids.
        """
        random_rows = np.random.choice(data_matrix.array_2d.shape[0], K, replace=False)
        centroids = data_matrix.array_2d[random_rows, :]
        return Matrix.from_array(centroids)

    @staticmethod
    def get_separation_within(data_matrix, centroids, S, K):
        """
        Calculates the within-cluster separation for each feature.

        Args:
            data_matrix (Matrix): The data matrix used for clustering.
            centroids (Matrix): The centroids of the clusters.
            S (Matrix): A matrix indicating the cluster assignments for each data point.
            K (int): The number of clusters.

        Returns:
            Matrix: A new Matrix containing the within-cluster separations.
        """
        separation_within = np.zeros((1, data_matrix.array_2d.shape[1]))
        
        # Iterate over each feature j
        for j in range(data_matrix.array_2d.shape[1]):
            # Calculate aj by summing over clusters (k) and data points (i)
            for k in range(1, K + 1):
                # Sum distances for all data points assigned to cluster k
                for i in range(data_matrix.array_2d.shape[0]):
                    # Check if data point i belongs to cluster k
                    if S.array_2d[i, 0] == k:
                        # Compute the Euclidean distance between the j-th feature of the i-th data point and the j-th feature of centroid k
                        distance = (data_matrix.array_2d[i, j] - centroids.array_2d[k - 1, j]) ** 2
                        separation_within[0, j] += distance
        
        # Return the result as a matrix object
        return Matrix.from_array(separation_within)
    
    @staticmethod
    def get_separation_between(data_matrix, centroids, S, K):
        """
        Calculates the between-cluster separation for each feature.

        Args:
            data_matrix (Matrix): The data matrix used for clustering.
            centroids (Matrix): The centroids of the clusters.
            S (Matrix): A matrix indicating the cluster assignments for each data point.
            K (int): The number of clusters.

        Returns:
            Matrix: A new Matrix containing the between-cluster separations.
        """
        separation_between = np.zeros((1, data_matrix.array_2d.shape[1]))

        # Compute the overall mean for each feature (D_j')
        overall_mean = data_matrix.array_2d.mean(axis=0)

        # Iterate over each feature (j)
        for j in range(data_matrix.array_2d.shape[1]):
            # Iterate over each cluster (k)
            for k in range(1, K + 1):
                # Get the points assigned to cluster k
                cluster_points = data_matrix.array_2d[S.array_2d.flatten() == k]
                N_k = len(cluster_points)  # Number of points in cluster k

                # Get the centroid value for the j-th feature of cluster k
                c_kj = centroids.array_2d[k - 1, j]

                # Calculate the distance between c_kj and the overall mean D_j'
                distance = np.linalg.norm(c_kj - overall_mean[j])

                # Update the separation for the j-th feature by adding N_k * distance
                separation_between[0, j] += N_k * distance

        # Return the result as a matrix object
        return Matrix.from_array(separation_between)

    @staticmethod
    def get_new_weights(data_matrix, centroids, old_weights, S, K):
        """
        Updates the weights based on within-cluster and between-cluster separations.

        Args:
            data_matrix (Matrix): The data matrix used for clustering.
            centroids (Matrix): The centroids of the clusters.
            old_weights (Matrix): The previous weights.
            S (Matrix): A matrix indicating the cluster assignments for each data point.
            K (int): The number of clusters.

        Returns:
            Matrix: A new Matrix containing the updated weights.
        """
        separation_within = Matrix.get_separation_within(data_matrix, centroids, S, K).array_2d.flatten()
        separation_between = Matrix.get_separation_between(data_matrix, centroids, S, K).array_2d.flatten()

        # Calculate the ratios b_j / a_j
        ratio = separation_between / separation_within

        # Normalize the ratio: (b_j / a_j) / sum(b_v / a_v)
        normalized_ratio = ratio / np.sum(ratio)

        # Convert old_weights to a NumPy array for addition
        old_weights_array = old_weights.array_2d.flatten()

        # Update the weights using the formula: w'_j = 1/2 * (w_j + normalized_ratio)
        new_weights = 0.5 * (old_weights_array + normalized_ratio)

        return Matrix.from_array(new_weights.reshape(1, -1))

    @staticmethod
    def get_groups(data_matrix, K):
        """
        Clusters the data matrix into K groups using a modified k-means algorithm.

        Args:
            data_matrix (Matrix): The data matrix to be clustered.
            K (int): The number of clusters.

        Returns:
            tuple: A tuple containing:
                - Matrix: The final cluster assignments for each data point.
                - Matrix: The final centroids for each cluster.
        """
        # Step 1: Initialize weights
        m = data_matrix.array_2d.shape[1]  # Number of features
        weights = Matrix.get_initial_weights(m)

        # Step 2: Get initial centroids
        centroids = Matrix.get_centroids(data_matrix, K)

        # Initialize cluster assignments (S)
        S = Matrix.from_array(np.zeros((data_matrix.array_2d.shape[0], 1)))

        # Iteratively update weights and centroids until convergence
        while True:
            # Step 3: Calculate distances and assign clusters
            distances = np.zeros((data_matrix.array_2d.shape[0], K))
            for k in range(K):
                distances[:, k] = data_matrix.get_weighted_distance(centroids, weights, k).array_2d.flatten()

            # Assign clusters based on the minimum distance
            S_new = Matrix.from_array(np.argmin(distances, axis=1).reshape(-1, 1))

            # Check for convergence (if cluster assignments do not change)
            if np.array_equal(S.array_2d, S_new.array_2d):
                break

            # Update cluster assignments
            S = S_new

            # Step 4: Update weights
            weights = Matrix.get_new_weights(data_matrix, centroids, weights, S, K)

            # Step 5: Update centroids based on current cluster assignments
            for k in range(K):
                centroids.array_2d[k] = data_matrix.array_2d[S.array_2d.flatten() == k].mean(axis=0)

        return S, centroids


# matrix = Matrix('Data.csv')
# matrix.standardize()
# clusters, centroids = Matrix.get_groups(matrix, K=3)
# print("Clusters:\n", clusters.array_2d)
# print("Centroids:\n", centroids.array_2d)
