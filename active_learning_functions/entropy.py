from .basemethod import BaseMethod
import numpy as np

class Entropy(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Predict class probabilities for each unlabeled sample
        prediction_probabilities = classifier.predict_proba(x_unlabeled)

        # Compute entropy for each sample
        epsilon = 1e-10  # to prevent log(0)
        entropy_scores = -np.sum(prediction_probabilities * np.log(prediction_probabilities + epsilon), axis=1)

        # Select top-k samples with highest entropy (most uncertain)
        selected_indices = np.argsort(-entropy_scores)[:budget]

        return selected_indices
