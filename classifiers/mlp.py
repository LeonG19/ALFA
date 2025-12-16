import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLPNet(nn.Module):
    """
    Simple feedforward neural network with variable hidden layers,
    and an `embed` method to extract penultimate features.
    """
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int):
        super(MLPNet, self).__init__()
        layers = []
        dims = [input_dim] + list(hidden_layers)
        # Hidden layers + ReLU
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        # Final linear layer
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the activations immediately before the final output layer.
        """
        # all layers except the last Linear
        return self.net[:-1](x)


class TorchMLPClassifier:
    """
    sklearn-like interface for an MLP classifier using PyTorch,
    with MC-dropout sampling and embedding extraction via DataLoader.
    """
    def __init__(
        self,
        cfg,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        max_iter: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        random_state: int | None = None,
        device: str | None = None,
    ):
        # cfg must have attributes:
        #   DATASET.VAL_BATCH_SIZE
        #   DATASET.NUM_WORKERS
        #   DATASET.NUM_CLASS
        self.cfg = cfg
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.device = torch.device(device) if device else \
                      torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MLPNet | None = None
        self.classes_: np.ndarray | None = None
        print(f"Using device: {self.device}")

    def _set_seed(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> "TorchMLPClassifier":
        """
        Train the MLP on data X, y using mini-batch gradient descent.
        """
        # convert inputs
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        self._set_seed()
        self.classes_ = torch.unique(y).cpu().numpy()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.DATASET.NUM_WORKERS)

        in_dim = X.shape[1]
        out_dim = len(self.classes_)
        self.model = MLPNet(in_dim, self.hidden_layer_sizes, out_dim).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.max_iter):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Return class predictions for X.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Return class probabilities for X.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def mc_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass over X with dropout active, returning probabilities.
        Batches through DataLoader for VAL_BATCH_SIZE.
        """
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.DATASET.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
        )
        self.model.train()  # keep dropout active

        predictions = []
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                preds = F.softmax(logits, dim=1)
                predictions.append(preds)
        return torch.cat(predictions, dim=0)

    def mc_sample(self, x_unlabeled: np.ndarray | torch.Tensor, trials: int) -> np.ndarray:
        """
        Perform MC Dropout sampling: returns array shape (N, trials, C).
        """
        if isinstance(x_unlabeled, np.ndarray):
            X = torch.from_numpy(x_unlabeled).float()
        else:
            X = x_unlabeled.float()
        X = X.to(self.device)

        # enable dropout but freeze batchnorm
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        N = X.shape[0]
        C = self.cfg.DATASET.NUM_CLASS
        probs = torch.zeros((N, trials, C), device=self.device)

        with torch.no_grad():
            for t in range(trials):
                probs[:, t, :] = self.mc_predict(X)
        return probs.cpu().numpy()

    def embed(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Extract penultimate-layer embeddings for X via DataLoader.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.DATASET.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
        )

        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_X = batch_X.to(self.device)
                feats = self.model.embed(batch_X)
                embeddings.append(feats.cpu().numpy())
        return np.concatenate(embeddings, axis=0)
