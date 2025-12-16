from .basemethod import BaseMethod
import numpy as np

class Margin(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Predict class probabilities for each unlabeled sample
        prediction_probabilities = classifier.predict_proba(x_unlabeled)

        # Ensure we have at least two class probabilities
        if prediction_probabilities.shape[1] < 2:
            raise ValueError("Margin sampling requires at least 2 class probabilities per sample.")

        # Get top 2 probabilities (descending)
        top2_probs = -np.sort(-prediction_probabilities, axis=1)[:, :2]

        # Compute margin
        margins = top2_probs[:, 0] - top2_probs[:, 1]

        # Select most uncertain (smallest margin)
        selected_indices = np.argsort(margins)[:budget]

        return selected_indices
