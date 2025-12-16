from .basemethod import BaseMethod
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

class Alpha(BaseMethod):
    def __init__(self, alpha_cap_step=0.03125):
        self.alpha_cap_step = alpha_cap_step

    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        device = classifier.device
        ulb_emb = classifier.embed(x_unlabeled)  # [N_u, D]
        ulb_emb_torch = torch.tensor(ulb_emb, dtype=torch.float32, device=device)

        ulb_preds = classifier.predict(x_unlabeled)  # [N_u]
        ulb_preds_torch = torch.tensor(ulb_preds, dtype=torch.long, device=device)

        grads = self._get_grads(ulb_emb_torch, ulb_preds_torch, classifier)

        # Get class anchors from labeled data
        lbl_preds = classifier.predict(x_labeled)
        lbl_emb = classifier.embed(x_labeled)
        n_classes = np.max(lbl_preds) + 1
        anchors = []
        for c in range(n_classes):
            if np.sum(lbl_preds == c) > 0:
                anchors.append(np.mean(lbl_emb[lbl_preds == c], axis=0))
            else:
                anchors.append(np.zeros(lbl_emb.shape[1]))
        anchors = np.stack(anchors)
        anchors_torch = torch.tensor(anchors, dtype=torch.float32, device=device)

        # Closed-form alpha mixing
        pred_change_mask = torch.zeros(len(ulb_emb), dtype=torch.bool, device=device)
        alpha_cap = 0.0
        while alpha_cap < 1.0:
            alpha_cap += self.alpha_cap_step
            changed, _ = self._find_changes(ulb_emb_torch, ulb_preds_torch, anchors_torch, grads, classifier, alpha_cap)
            pred_change_mask |= changed
            if pred_change_mask.sum().item() >= budget:
                break

        if pred_change_mask.sum().item() == 0:
            # fallback to random
            return np.random.choice(np.arange(len(x_unlabeled)), size=budget, replace=False)

        # Select top-k via KMeans
        changed_emb = ulb_emb[pred_change_mask.cpu().numpy()]
        selected = self._kmeans_select(changed_emb, budget)
        selected_idxs = np.where(pred_change_mask.cpu().numpy())[0][selected]
        return selected_idxs

    def _get_grads(self, emb, preds, classifier):
        emb = emb.clone().detach().requires_grad_(True)
        logits = classifier.predict_logits(emb, from_emb=True)
        loss = F.cross_entropy(logits, preds)
        grads = torch.autograd.grad(loss, emb)[0]
        return grads.detach()

    def _find_changes(self, ulb_emb, ulb_preds, anchors, grads, classifier, alpha_cap):
        n, d = ulb_emb.shape
        alpha_cap /= np.sqrt(d)
        pred_change = torch.zeros(n, dtype=torch.bool, device=ulb_emb.device)

        for c in range(anchors.shape[0]):
            anchor = anchors[c].reshape(1, -1).repeat(n, 1)
            z = anchor - ulb_emb
            alpha = (alpha_cap * z.norm(dim=1) / (grads.norm(dim=1) + 1e-8)).unsqueeze(1) * grads / (z + 1e-8)
            alpha = torch.clamp(alpha, min=1e-8, max=alpha_cap)

            mixed = (1 - alpha) * ulb_emb + alpha * anchor
            logits = classifier.predict_logits(mixed, from_emb=True)
            new_preds = logits.argmax(dim=1)
            pred_change |= (new_preds != ulb_preds)

        return pred_change, alpha

    def _kmeans_select(self, feats, k):
        if len(feats) <= k:
            return np.arange(len(feats))
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(feats)
        dists = np.linalg.norm(feats - km.cluster_centers_[km.labels_], axis=1)
        selected = []
        for i in range(k):
            cluster_points = np.where(km.labels_ == i)[0]
            if len(cluster_points) == 0:
                continue
            center = km.cluster_centers_[i]
            point = cluster_points[np.argmin(np.linalg.norm(feats[cluster_points] - center, axis=1))]
            selected.append(point)
        return np.array(selected)
