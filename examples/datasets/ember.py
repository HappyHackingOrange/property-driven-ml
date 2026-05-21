"""
EMBER2024 dataset wrapper for property-driven training.

EMBER2024 (Joyce et al., KDD 2025) is a public malware-classification
benchmark with pre-extracted numerical features for ~3.2M files across six
formats (Win32, Win64, .NET, APK, ELF, PDF). It supports multiple
classification tasks (malicious/benign, family, MITRE ATT&CK behavior),
ships temporal train/test splits (52 weeks / 12 weeks) for drift evaluation,
and includes a challenge set of ~6,315 evasive samples.

Loading goes through the official ``thrember`` package, which abstracts
over the HuggingFace Parquet / NPZ files and produces numpy ndarrays.
Install:

    pip install git+https://github.com/FutureComputing4AI/EMBER2024

The dataset is large (tens of GB downloaded, ~30 GB vectorized for the
training split alone), so this module does not load it into memory.
Instead, ``EmberDataset`` wraps the on-disk ndarrays (typically memmapped
by thrember) and applies z-score normalization per-sample at access time.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from property_driven_ml.training.mode import Mode
from examples.models import EmberNet


# Default on-disk location for the vectorized EMBER2024 features. Sits
# outside the repo so it survives clean clones, and outside ~/.cache so
# it's discoverable.
DEFAULT_DATA_DIR = os.path.expanduser("~/data/ember2024")


class EmberDataset(torch.utils.data.Dataset):
    """Wraps an EMBER2024 numpy array (typically memmapped by thrember)
    with per-sample z-score normalization. Avoids loading the full dataset
    into RAM; the per-feature ``mean`` and ``std`` are small (length =
    feature dim) and live in memory.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.X = X
        self.y = y
        self.mean = mean.astype(np.float32)
        # Avoid divide-by-zero on constant features.
        self.std = np.where(std == 0.0, 1.0, std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = (self.X[idx].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x), int(self.y[idx])


def _filter_labeled_binary(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """EMBER encodes unlabeled samples as -1 in the binary malicious/benign
    label. Drop those rows for binary classification."""
    mask = (y == 0) | (y == 1)
    return X[mask], y[mask].astype(np.int64)


def _ensure_dataset_ready(data_dir: str, download_if_missing: bool) -> None:
    """Make sure ``data_dir`` contains vectorized EMBER2024 features. If
    ``download_if_missing`` is True and the directory looks empty, download
    and vectorize via thrember. Otherwise raise FileNotFoundError."""
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        return

    if not download_if_missing:
        raise FileNotFoundError(
            f"Vectorized EMBER2024 features not found at {data_dir}. "
            "Either set download_if_missing=True (this downloads several "
            "tens of GB and then vectorizes, taking a long time), or run\n"
            "    import thrember\n"
            "    thrember.download_dataset(data_dir)\n"
            "    thrember.create_vectorized_features(data_dir)\n"
            "manually first."
        )

    import thrember  # lazy: only needed when actually downloading

    os.makedirs(data_dir, exist_ok=True)
    thrember.download_dataset(data_dir)
    thrember.create_vectorized_features(data_dir)


def create_ember_datasets(
    batch_size: int,
    data_dir: Optional[str] = None,
    task: str = "malicious",
    download_if_missing: bool = False,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    Tuple[Tuple[float, ...], Tuple[float, ...]],
    Mode,
]:
    """
    Create EMBER2024 train/test loaders via the official ``thrember`` package.

    Args:
        batch_size: Size of training batches.
        data_dir: Directory where vectorized EMBER2024 features live on disk.
            Defaults to ``~/data/ember2024``. If ``download_if_missing`` is
            True and the directory is empty, the dataset is downloaded and
            vectorized into this directory.
        task: Currently only ``"malicious"`` (binary malicious/benign) is
            implemented. ``"family"`` (multi-class over 6,787 families) and
            ``"behavior"`` (multi-label MITRE ATT&CK) will raise
            NotImplementedError.
        download_if_missing: If True, fetch the dataset and vectorize when
            the directory is empty. Off by default because the download
            and vectorization together take significant time and disk.
        max_samples: If set, cap the training split to this many samples
            and the test split proportionally. Useful for fast iteration
            on machines that can't hold all of EMBER2024 in RAM.
        seed: RNG seed used when subsampling.

    Returns:
        (train_loader, test_loader, model, (mean, std), mode) following the
        same shape as the other dataset creators in this package.

    Notes:
        Features are z-score normalized using the training-split statistics
        only. Normalization is applied per-sample inside ``EmberDataset``,
        so the full array never needs to be materialized as a single tensor.
        The returned ``(mean, std)`` tuple is ``(0.0,)/(1.0,)`` because the
        normalization is already applied by the Dataset; this matches the
        Alsomitra precedent and means an EpsilonBall ``epsilon`` is
        interpreted as already in standardized units.

        Requires the ``thrember`` package:

            pip install git+https://github.com/FutureComputing4AI/EMBER2024
    """
    if task != "malicious":
        raise NotImplementedError(
            f"Task '{task}' not yet supported. Only 'malicious' (binary "
            "malicious/benign) is implemented. The 'family' (multi-class "
            "over 6,787 families) and 'behavior' (multi-label MITRE ATT&CK) "
            "variants would require a different label-handling path and a "
            "different Mode; see the EMBER2024 paper (Joyce et al., KDD "
            "2025) for the label schema."
        )

    try:
        import thrember
    except ImportError as e:
        raise ImportError(
            "thrember is required to load EMBER2024. Install with:\n"
            "    pip install git+https://github.com/FutureComputing4AI/EMBER2024\n"
            "For pipeline testing without EMBER, use "
            "create_synthetic_tabular_datasets instead."
        ) from e

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    _ensure_dataset_ready(data_dir, download_if_missing)

    X_train, y_train = thrember.read_vectorized_features(data_dir, subset="train")
    X_test, y_test = thrember.read_vectorized_features(data_dir, subset="test")

    X_train, y_train = _filter_labeled_binary(X_train, y_train)
    X_test, y_test = _filter_labeled_binary(X_test, y_test)

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        n_train = min(max_samples, len(X_train))
        n_test = min(max(max_samples // 5, 1), len(X_test))
        train_idx = rng.choice(len(X_train), size=n_train, replace=False)
        test_idx = rng.choice(len(X_test), size=n_test, replace=False)
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        X_test, y_test = X_test[test_idx], y_test[test_idx]

    # Compute z-score statistics on the training split only. Works on
    # memmaps by streaming internally in numpy; slow for the full split
    # but only happens once per process.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    train_ds = EmberDataset(X_train, y_train, mean, std)
    test_ds = EmberDataset(X_test, y_test, mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[1]
    model = EmberNet(input_dim=n_features, n_classes=2)

    return (
        train_loader,
        test_loader,
        model,
        ((0.0,), (1.0,)),
        Mode.MultiClassClassification,
    )
