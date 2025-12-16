from .basemethod import BaseMethod
import numpy as np

class Random(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        self.rng = np.random.RandomState(2)
        # Get the indices of the unlabeled pool
        n_samples = len(x_unlabeled)

        # Randomly choose 'budget' indices without replacement
        random_indices = self.rng.choice(n_samples, size=budget, replace=False)

        return random_indices
