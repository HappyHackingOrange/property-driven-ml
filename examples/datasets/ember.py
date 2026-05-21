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
    """Wraps an EMBER2024 feature memmap with per-sample z-score normalization
    via an index array. Never materializes the full feature matrix:

    - ``X`` is a memmap (or memmap-like ndarray) of shape ``(N_total, dim)``.
    - ``y`` is an in-memory int array of shape ``(N_total,)`` (small).
    - ``indices`` selects which rows of ``X`` / ``y`` are actually used
      (labeled-binary filter plus any subsampling).

    PyTorch's DataLoader reads samples one at a time, so the memmap only
    touches the row pages it needs per batch.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.X = X
        self.y = y
        self.indices = indices
        self.mean = mean.astype(np.float32)
        # Avoid divide-by-zero on constant features.
        self.std = np.where(std == 0.0, 1.0, std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = (self.X[real_idx].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x), int(self.y[real_idx])


def _open_memmap(
    data_dir: str, subset: str, ndim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Open the ``X_{subset}.dat`` and ``y_{subset}.dat`` memmaps thrember
    writes during vectorization, without materializing them in RAM.

    Equivalent to ``thrember.read_vectorized_features`` minus its
    ``np.array(X)`` call that forces the data into memory.
    """
    X_path = os.path.join(data_dir, f"X_{subset}.dat")
    y_path = os.path.join(data_dir, f"y_{subset}.dat")
    X = np.memmap(X_path, dtype=np.float32, mode="r").reshape(-1, ndim)
    # y is small (one int32 per row) and we need it for filtering and
    # subsampling, so materialize it explicitly. Copy out of the memmap to
    # avoid surprises if the file is closed later.
    y = np.array(np.memmap(y_path, dtype=np.int32, mode="r"))
    return X, y


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
        from thrember import PEFeatureExtractor
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

    # Get feature dim from thrember's extractor; current value is 2568 for
    # EMBER2024 feature version 3, but reading from the extractor keeps us
    # robust to future feature-set updates.
    ndim = PEFeatureExtractor().dim

    # Open the disk memmaps directly. We deliberately do NOT use
    # thrember.read_vectorized_features here because that wraps the open
    # in np.array(...).reshape(...), which materializes ~54 GB for the
    # train split. Reading the memmap and reshaping returns a view, not
    # a copy, so RAM use stays bounded by the batch size.
    X_train, y_train = _open_memmap(data_dir, "train", ndim)
    X_test, y_test = _open_memmap(data_dir, "test", ndim)

    # Label-filter for the binary task: keep rows where y is 0 or 1.
    # Filtering is done on indices, not on X, so we never materialize.
    train_labeled = np.where((y_train == 0) | (y_train == 1))[0]
    test_labeled = np.where((y_test == 0) | (y_test == 1))[0]

    # Optional subsampling.
    rng = np.random.default_rng(seed)
    if max_samples is not None:
        n_train = min(max_samples, len(train_labeled))
        n_test = min(max(max_samples // 5, 1), len(test_labeled))
        train_idx = rng.choice(train_labeled, size=n_train, replace=False)
        test_idx = rng.choice(test_labeled, size=n_test, replace=False)
    else:
        train_idx = train_labeled
        test_idx = test_labeled

    # Compute z-score statistics from a bounded sample of the training rows
    # we actually plan to use. Reading the full ~50 GB matrix to compute
    # exact stats would defeat the purpose of memmapping. 100k rows is
    # plenty for stable per-feature mean/std on this dataset.
    n_stats = min(len(train_idx), 100_000)
    if len(train_idx) > n_stats:
        stats_idx = rng.choice(train_idx, size=n_stats, replace=False)
    else:
        stats_idx = train_idx
    # Sort indices so the memmap reads contiguously where possible.
    stats_sample = X_train[np.sort(stats_idx)]
    mean = stats_sample.mean(axis=0).astype(np.float32)
    std = stats_sample.std(axis=0).astype(np.float32)

    # y as int64 once, in RAM (small).
    y_train_i64 = y_train.astype(np.int64)
    y_test_i64 = y_test.astype(np.int64)

    train_ds = EmberDataset(X_train, y_train_i64, train_idx, mean, std)
    test_ds = EmberDataset(X_test, y_test_i64, test_idx, mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = EmberNet(input_dim=ndim, n_classes=2)

    return (
        train_loader,
        test_loader,
        model,
        ((0.0,), (1.0,)),
        Mode.MultiClassClassification,
    )
