"""
Synthetic tabular dataset for validating the property-driven-ml pipeline
on tabular numerical features before plugging in a real dataset (EMBER).

The dataset is two well-separated Gaussian clusters in a feature space of
configurable dimension. Class 0 is "benign", class 1 is "malicious". The
shape mimics what EMBER2024 features look like (numerical, pre-extracted)
without the download/loading complexity, so the pipeline can be exercised
end-to-end before swapping in real data.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from property_driven_ml.training.mode import Mode
from examples.models import EmberNet


def _make_gaussian_clusters(
    n_samples: int,
    n_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-class Gaussian clusters, mean-centered to unit-ish scale."""
    rng = np.random.default_rng(seed)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # Class centers separated by ~2 sigma along every dimension.
    center = np.full(n_features, 1.0, dtype=np.float32)
    X_pos = rng.normal(loc=center, scale=1.0, size=(n_pos, n_features)).astype(np.float32)
    X_neg = rng.normal(loc=-center, scale=1.0, size=(n_neg, n_features)).astype(np.float32)
    y_pos = np.ones(n_pos, dtype=np.int64)
    y_neg = np.zeros(n_neg, dtype=np.int64)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def create_synthetic_tabular_datasets(
    batch_size: int,
    n_features: int = 64,
    n_train: int = 4000,
    n_test: int = 1000,
    seed: int = 0,
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    Tuple[Tuple[float, ...], Tuple[float, ...]],
    Mode,
]:
    """
    Create train/test loaders for a synthetic tabular binary-classification task.

    Returns the same five-tuple shape as the other dataset creators:
        (train_loader, test_loader, model, (mean, std), mode)

    The mean/std tuple is (0.0,)/(1.0,) because the synthetic data is already
    on a unit-ish scale. For real EMBER, these would be the per-feature
    train-split statistics used to standardize the inputs.
    """
    X_train, y_train = _make_gaussian_clusters(n_train, n_features, seed=seed)
    X_test, y_test = _make_gaussian_clusters(n_test, n_features, seed=seed + 1)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = EmberNet(input_dim=n_features, n_classes=2)
    mean, std = (0.0,), (1.0,)

    return train_loader, test_loader, model, (mean, std), Mode.MultiClassClassification
