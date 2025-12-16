from .basemethod import BaseMethod
import numpy as np

class LowerLimit(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        print("we are here")
        return np.array([], dtype=int)
