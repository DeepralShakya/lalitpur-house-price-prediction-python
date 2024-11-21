import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters, max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)
        random_indices = np.random.permutation(n_samples)[:self.n_clusters]
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        # Corrected distance computation
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return distances
