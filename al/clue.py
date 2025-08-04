from .basemethod import BaseMethod
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

class CLUE(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Get penultimate-layer embeddings
        penultimate_embeddings = classifier.embed(x_unlabeled)  # shape: [N, D]

        # Get class probabilities
        probs = classifier.predict_proba(x_unlabeled)  # shape: [N, C]

        # Compute uncertainty via entropy
        sample_weights = entropy(probs.T)  # scipy expects transposed input

        # Run weighted KMeans over embeddings
        km = KMeans(n_clusters=budget, random_state=42)
        km.fit(penultimate_embeddings, sample_weight=sample_weights)

        # Find closest points to cluster centers
        dists = pairwise_distances(km.cluster_centers_, penultimate_embeddings)
        sort_idxs = dists.argsort(axis=1)

        # Select one point per cluster (with deduplication)
        q_idxs = []
        ax, rem = 0, budget
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))  # remove duplicates
            rem = budget - len(q_idxs)
            ax += 1

        return np.array(q_idxs)
