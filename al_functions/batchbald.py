from .basemethod import BaseMethod
import numpy as np
import torch
from batchbald_redux.batchbald import get_batchbald_batch

class BatchBALD(BaseMethod):
    def __init__(self):
        super().__init__()
        self.dropout_trials = 25
        self.num_samples = 100

    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        mc_output = classifier.mc_sample(x_unlabeled, self.dropout_trials)
        result = get_batchbald_batch(mc_output, budget, self.num_samples)
        return np.array(result.indices)
