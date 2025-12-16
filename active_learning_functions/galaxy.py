from .basemethod import BaseMethod
import numpy as np
from scipy.spatial.distance import cdist


class Galaxy(BaseMethod):
    def _hac(self, clusters, c_dists):
        num_empty = 0
        while c_dists.shape[0] / float(len(clusters) - num_empty) < 2:
            num_empty += 1
            num_elem = np.array([float(len(c)) for c in clusters])
            i, j = np.unravel_index(np.argmin(c_dists), c_dists.shape)
            assert num_elem[i] != 0. and num_elem[j] != 0.
            c_dists[i] = (c_dists[i] * num_elem[i] + c_dists[j] * num_elem[j]) / (num_elem[i] + num_elem[j])
            c_dists[:, i] = (c_dists[:, i] * num_elem[i] + c_dists[:, j] * num_elem[j]) / (num_elem[i] + num_elem[j])
            c_dists[j] = float("inf")
            c_dists[:, j] = float("inf")
            clusters[i] = clusters[i] + clusters[j]
            clusters[j] = []
        new_clusters = [c for c in clusters if len(c) != 0]
        return new_clusters

    def _cluster(self, features):
        np.random.seed(12345)
        subset_idx = np.random.permutation(len(features))[:1000]
        features_subset = features[subset_idx]

        # Compute distances only for the subset
        dist_subset = cdist(features_subset, features_subset)
        np.fill_diagonal(dist_subset, float('inf'))

        clusters = self._hac([[i] for i in range(len(subset_idx))], dist_subset)
        centers = [np.mean(features_subset[c], axis=0) for c in clusters]
        centers = np.stack(centers)

        # Assign all features to nearest cluster center
        cluster_assignments = np.argmin(cdist(features, centers), axis=1)
        return cluster_assignments

    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Step 1: Embed the unlabeled pool
        embeddings = classifier.embed(x_unlabeled)
        pred_probs = classifier.predict_proba(x_unlabeled)

        # Step 2: Get uncertainty (margin) scores
        sorted_probs = -np.sort(-pred_probs, axis=1)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        uncertain_order = np.argsort(margins)

        # Step 3: Clustering
        cluster_ids = self._cluster(embeddings)

        # Step 4: Build per-cluster queues from most uncertain samples
        clusters = [[] for _ in range(np.max(cluster_ids) + 1)]
        cluster_batch_size = int(1.25 * budget)
        added = 0
        for idx in uncertain_order:
            clusters[cluster_ids[idx]].append(idx)
            added += 1
            if added == cluster_batch_size:
                break

        # Step 5: Round-robin pick from clusters
        chosen = []
        cluster_order = np.argsort([len(c) for c in clusters])
        c_idx = 0
        while len(chosen) < budget:
            cluster_i = cluster_order[c_idx]
            if clusters[cluster_i]:
                chosen.append(clusters[cluster_i].pop(0))
            c_idx = (c_idx + 1) % len(clusters)

        return np.array(chosen)
