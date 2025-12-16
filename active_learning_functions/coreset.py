from .basemethod import BaseMethod
import numpy as np
import torch
from sklearn.metrics import pairwise_distances

class Coreset(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        with torch.no_grad():
            # Get embeddings for both labeled and unlabeled data
            unlabeled_embeddings = classifier.embed(x_unlabeled)
            labeled_embeddings = classifier.embed(x_labeled)

            print(labeled_embeddings.shape)

            # Compute distances from each unlabeled point to the closest labeled point
            distances = pairwise_distances(unlabeled_embeddings, labeled_embeddings, metric='euclidean')
            min_distances = np.min(distances, axis=1)

            # Select top-k farthest points (largest minimum distances)
            selected_indices = np.argsort(-min_distances)[:budget]

        return selected_indices
