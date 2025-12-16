from .basemethod import BaseMethod
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
def fit_gmm(X, n_components=9):
       """
       Fit a Gaussian Mixture Model to data X and return the trained GMM.
       """
       gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=1)
       gmm.fit(X)
       return gmm

def compute_informativeness_scores(X_unlabeled, gm_labeled, gm_unlabeled, budget):
       """
       Compute the informativeness score for each sample in X_unlabeled.
       IScore = log P(x | GMM_unlabeled) - log P(x | GMM_labeled)
       """
       # Compute log-likelihoods of each sample under both GMMs
       log_lik_ul = gm_unlabeled.score_samples(X_unlabeled)  # log P(x | ψ_UL)
       log_lik_l  = gm_labeled.score_samples(X_unlabeled)    # log P(x | ψ_L)
       # Informativeness score for each sample (difference of log-likelihoods)
       iscore = log_lik_ul - log_lik_l
       sorted_scores = np.argsort(iscore)[-int(budget):]
       print(len(sorted_scores))
       return sorted_scores

class Density(BaseMethod):
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
         '''
         scaler = StandardScaler().fit(x_labeled)
         x_labeled = scaler.transform(x_labeled)
         x_unlabeled = scaler.transform(x_unlabeled)
         '''
         gmm_2017 = fit_gmm(x_unlabeled)
         gmm_2018 = fit_gmm(x_labeled)
         return compute_informativeness_scores(x_unlabeled, gmm_2017, gmm_2018, budget)
