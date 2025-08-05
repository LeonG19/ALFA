from .basemethod import BaseMethod
import numpy as np
from sklearn.mixture import GaussianMixture

class DiaNA(BaseMethod):
    def __init__(self):
        super().__init__()
        self.confidence_threshold = 0.95
        self.topk = 32

    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        # Get embeddings and predictions
        features = classifier.embed(x_unlabeled)  # [N, D]
        probs = classifier.predict_proba(x_unlabeled)  # [N, C]
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # Compute class centroids from labeled set
        labeled_features = classifier.embed(x_labeled)
        labeled_labels = classifier.predict(x_labeled) if len(x_labeled) > 0 else np.array([])
        if len(labeled_labels) == 0:
            raise ValueError("DiaNA requires some labeled data to compute centroids.")

        num_classes = probs.shape[1]
        centroids = []
        for c in range(num_classes):
            class_feats = labeled_features[labeled_labels == c]
            if len(class_feats) > 0:
                centroid = np.mean(class_feats, axis=0)
            else:
                centroid = np.zeros(features.shape[1])
            centroids.append(centroid)
        centroids = np.stack(centroids)  # [C, D]

        # Similarity-based labels using top-k IoU of feature indices
        def topk_indices(v, k):
            return np.argsort(v)[-k:]

        sim_labels = []
        for feat in features:
            feat_topk = set(topk_indices(feat, self.topk))
            best_c = -1
            best_iou = -1
            for c, center in enumerate(centroids):
                center_topk = set(topk_indices(center, self.topk))
                iou = len(feat_topk & center_topk) / len(feat_topk | center_topk)
                if iou > best_iou:
                    best_iou = iou
                    best_c = c
            sim_labels.append(best_c)
        sim_labels = np.array(sim_labels)

        # Compute informativeness score: negative log prob of similarity-based label
        info_scores = -np.log(np.clip(probs[np.arange(len(probs)), sim_labels], 1e-12, 1.0))

        # Categorize samples (UI = uncertain-inconsistent)
        is_confident = confidences >= self.confidence_threshold
        is_consistent = (preds == sim_labels)

        categories = np.zeros(len(x_unlabeled))  # 0: UI, 1: UC, 2: CI, 3: CC
        categories[~is_confident & ~is_consistent] = 0  # UI
        categories[~is_confident & is_consistent] = 1   # UC
        categories[is_confident & ~is_consistent] = 2   # CI
        categories[is_confident & is_consistent] = 3    # CC

        # Normalize info scores for GMM
        scores = (info_scores - info_scores.min()) / (info_scores.max() - info_scores.min() + 1e-8)

        # Prepare labeled scores and category labels for GMM training
        labeled_scores = []
        labeled_cats = []
        for c in range(4):
            idxs = np.where(categories == c)[0]
            labeled_scores += list(scores[idxs])
            labeled_cats += [c] * len(idxs)
        labeled_cats = np.array(labeled_cats)

        # Train semi-supervised GMM
        gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
        gmm.fit(labeled_scores)

        # Predict component probabilities
        unlab_scores = scores.reshape(-1, 1)
        probs_gmm = gmm.predict_proba(unlab_scores)

        # Select top-k from component corresponding to UI
        ui_component = 0  # We assume UI maps to component 0
        ui_probs = probs_gmm[:, ui_component]
        top_ui = np.argsort(-ui_probs)[:budget]

        return top_ui
