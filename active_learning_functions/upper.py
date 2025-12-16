from .basemethod import BaseMethod
import numpy as np

class UpperLimit(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Get the indices of the unlabeled pool
        n_samples = len(x_unlabeled)

        # Randomly choose 'budget' indices without replacement
        random_indices = np.random.choice(n_samples, size=n_samples, replace=False)

        return random_indices
