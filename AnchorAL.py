# anchoral_min.py
# Compact AnchorAL for tabular data (e.g., CIC-IDS)
#
# Provides a class `anchoral` with:
#   - create_representations(X)
#   - extract_anchors(X_l, y_l, classes)
#   - extract_knn(Z_query, pool_idx, k)
#   - subset(X, y, labeled_idx, unlabeled_idx)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus
from sklearn.neighbors import NearestNeighbors

class anchoral:
    def __init__(
        self,
        A: int = 10,          # anchors per class
        K: int = 50,          # neighbors per anchor
        M: int = 1000,        # max subpool size
        rep: str = "scaled",  # "scaled" or "pca"
        pca_dim: int = 64,
        metric: str = "cosine",  # "cosine" or "euclidean"
        random_state: int = 42,
    ):
        self.A = A
        self.K = K
        self.M = M
        self.rep = rep
        self.pca_dim = pca_dim
        self.metric = metric
        self.rng = np.random.RandomState(random_state)

        self.scaler = None
        self.pca = None
        self.Z_all = None

    # -------- representations --------
    def create_representations(self, X: np.ndarray) -> np.ndarray:
        """
        Fit representation on X and return Z (representations).
        - Standard scaling
        - Optional PCA
        - Normalize if metric == "cosine"
        """
        X = np.asarray(X)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        if self.rep == "pca":
            d = min(self.pca_dim, Xs.shape[1])
            self.pca = PCA(n_components=d, random_state=self.rng.randint(1, 1_000_000))
            Z = self.pca.fit_transform(Xs)
        else:
            Z = Xs

        if self.metric == "cosine":
            norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
            Z = Z / norms

        self.Z_all = Z
        return Z

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X) if self.scaler else X
        Z = self.pca.transform(Xs) if self.pca else Xs
        if self.metric == "cosine":
            norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
            Z = Z / norms
        return Z

    # -------- anchors --------
    def extract_anchors(self, X_l: np.ndarray, y_l: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Select A anchors per class using k-means++ in representation space.
        Returns indices relative to X_l.
        """
        Z_l = self._transform(X_l)
        anchors = []
        for c in classes:
            mask = (y_l == c)
            Zc = Z_l[mask]
            idx_c = np.where(mask)[0]
            if Zc.shape[0] == 0:
                continue
            if Zc.shape[0] <= self.A:
                anchors.extend(idx_c.tolist())
            else:
                _, seed_idx = kmeans_plusplus(Zc, n_clusters=self.A, random_state=self.rng)
                anchors.extend(idx_c[seed_idx].tolist())
        return np.asarray(anchors, dtype=int)

    # -------- kNN --------
    def extract_knn(self, Z_query: np.ndarray, pool_idx: np.ndarray, k: int):
        """
        Return k nearest neighbors of Z_query within pool_idx.
        Returns (indices, similarity_scores).
        """
        pool_idx = np.asarray(pool_idx, dtype=int)
        Z_pool = self.Z_all[pool_idx]
        nn = NearestNeighbors(metric="cosine" if self.metric=="cosine" else "euclidean")
        nn.fit(Z_pool)
        dists, idx_local = nn.kneighbors(Z_query[None, :], n_neighbors=min(k, len(pool_idx)))
        dists, idx_local = dists[0], idx_local[0]
        sims = 1.0 - dists if self.metric == "cosine" else -dists
        return pool_idx[idx_local], sims

    # -------- main --------
    def subset(self, X: np.ndarray, y: np.ndarray, labeled_idx: np.ndarray, unlabeled_idx: np.ndarray) -> np.ndarray:
        """
        Return subpool indices of size up to M (global indices).
        """
        labeled_idx = np.asarray(labeled_idx, dtype=int)
        unlabeled_idx = np.asarray(unlabeled_idx, dtype=int)
        classes = np.unique(y[labeled_idx])

        anchors_local = self.extract_anchors(X[labeled_idx], y[labeled_idx], classes)
        anchor_global = labeled_idx[anchors_local] if anchors_local.size else labeled_idx

        scores = {}
        for a in anchor_global:
            Zq = self.Z_all[a]
            nn_idx, nn_sim = self.extract_knn(Zq, unlabeled_idx, self.K)
            for i, s in zip(nn_idx, nn_sim):
                scores.setdefault(int(i), []).append(float(s))

        if not scores:
            m = min(self.M, len(unlabeled_idx))
            return self.rng.choice(unlabeled_idx, size=m, replace=False)

        items = np.array([(i, np.mean(v)) for i, v in scores.items()], dtype=object)
        order = np.argsort([-x for x in items[:,1].astype(float)])
        top = items[order][: min(self.M, items.shape[0])]
        return top[:,0].astype(int)
