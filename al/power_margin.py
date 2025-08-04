import numpy as np
from .basemethod import BaseMethod

class PowerMargin(BaseMethod):
    def __init__(self, epsilon=1e-10, random_state=2):
        super().__init__()
        self.epsilon = epsilon  # To avoid log(0)
        self.rng = np.random.RandomState(random_state)

    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Predict class probabilities for each unlabeled sample
        prediction_probabilities = classifier.predict_proba(x_unlabeled)

        print("power margin prediction_probabilities", prediction_probabilities)

        # Check at least 2 classes
        if prediction_probabilities.shape[1] < 2:
            raise ValueError("Margin sampling requires at least 2 class probabilities.")

        # Get top 2 class probabilities
        top2_probs = -np.sort(-prediction_probabilities, axis=1)[:, :2]

        # Compute margin = p1 - p2
        margins = top2_probs[:, 0] - top2_probs[:, 1]

        # Invert margin to get uncertainty score (smaller margin = higher uncertainty)
        scores = 1.0 / (margins + self.epsilon)

        # Apply PowerGumbel trick: log(score) + Gumbel noise
        gumbel_noise = self.rng.gumbel(loc=0, scale=1, size=scores.shape)
        power_scores = np.log(scores + self.epsilon) + gumbel_noise

        # Select top-k indices based on PowerGumbel scores
        selected_indices = np.argsort(-power_scores)[:budget]

        return selected_indices
