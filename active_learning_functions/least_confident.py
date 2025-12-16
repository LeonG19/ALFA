from .basemethod import BaseMethod
import numpy as np

class LeastConfident(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        prediction_probabilities = classifier.predict_proba(x_unlabeled)

        # Get the probabilities of the predicted class
        max_probs = np.max(prediction_probabilities, axis=1)

        # Get indices of samples with the least probability of the predicted class
        least_confident_indices = np.argsort(max_probs)[:budget]

        return least_confident_indices
