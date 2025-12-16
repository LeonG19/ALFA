from .basemethod import BaseMethod
import numpy as np

class EADA(BaseMethod):
    def __init__(self, energy_beta=1.0, first_sample_ratio=0.5):
        super().__init__()
        self.energy_beta = energy_beta
        self.first_sample_ratio = first_sample_ratio

    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Step 1: Get energy outputs from model (logits or energy scores)
        logits = classifier.predict_energy(x_unlabeled)  # shape: [N, C]
        logits = np.array(logits)

        # Step 2: Compute free energy: F(x) = -β * logsumexp(-logits / β)
        scaled_logits = -1.0 * logits / self.energy_beta
        logsumexp = np.log(np.sum(np.exp(scaled_logits), axis=1))  # shape: [N]
        free_energy = -1.0 * self.energy_beta * logsumexp  # shape: [N]

        # Step 3: Compute Min-vs-Second-Min energy difference (MvSM)
        # Smaller values → more uncertain
        top2 = np.sort(logits, axis=1)[:, :2]  # smallest two values
        mvsm = top2[:, 0] - top2[:, 1]  # shape: [N]

        # Step 4: Two-stage sampling
        totality = len(x_unlabeled)
        first_sample_num = int(np.ceil(totality * self.first_sample_ratio))
        second_sample_num = int(np.ceil(first_sample_num * (budget / first_sample_num)))
        second_sample_num = min(budget, second_sample_num)

        # Step 4a: Select top-k by free energy
        first_stage_indices = np.argsort(-free_energy)[:first_sample_num]

        # Step 4b: From those, select top-k by MvSM uncertainty
        candidate_mvsm = mvsm[first_stage_indices]
        second_stage_indices = first_stage_indices[np.argsort(-candidate_mvsm)[:second_sample_num]]

        return np.array(second_stage_indices)
