"""
EMBER2024 dataset wrapper for property-driven training.

EMBER2024 (Joyce et al., KDD 2025) is a public malware-classification
benchmark with pre-extracted numerical features for ~3.2M files across six
formats (Win32, Win64, .NET, APK, ELF, PDF). It supports multiple
classification tasks (malicious/benign, family, MITRE ATT&CK behavior),
ships temporal train/test splits (52 weeks / 12 weeks) for drift evaluation,
and includes a challenge set of ~6,300 evasive samples.

This module is currently a stub. The dataset is hosted on HuggingFace and
loaded via the `datasets` library, which is not yet a dependency. See the
implementation TODOs below.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple

from property_driven_ml.training.mode import Mode
from examples.models import EmberNet


# EMBER2024's pre-extracted feature vector dimension.
EMBER2024_FEATURE_DIM = 2381


def create_ember_datasets(
    batch_size: int,
    task: str = "malicious",  # or "family", "behavior"
    file_format: str = "Win32",  # or "Win64", ".NET", "APK", "ELF", "PDF"
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    Tuple[Tuple[float, ...], Tuple[float, ...]],
    Mode,
]:
    """
    Create EMBER2024 train/test loaders.

    Returns the same five-tuple shape as the other dataset creators:
        (train_loader, test_loader, model, (mean, std), mode)

    Args:
        batch_size: Size of training batches.
        task: Which classification task. "malicious" is binary
            malware/benign; "family" is multi-class over 6,787 families;
            "behavior" is multi-label over MITRE ATT&CK techniques.
        file_format: Which subset of EMBER2024 to load.

    NOT YET IMPLEMENTED. The following steps need to be filled in:

      1. Add `datasets` (HuggingFace) to project dependencies.
      2. Load the EMBER2024 split via:
             from datasets import load_dataset
             ds = load_dataset("joyce8/EMBER2024", file_format)
      3. Extract numerical feature vectors (Parquet / NPZ formats supplied
         with the dataset). Each sample is a length-EMBER2024_FEATURE_DIM
         float vector.
      4. Convert string/categorical labels to ints (for "family") or 0/1
         (for "malicious") or multi-hot (for "behavior").
      5. Compute per-feature mean and std on the training split only and
         apply z-score normalization to both splits. This matters: hyper-
         rectangle constraints will be specified in standardized units.
      6. Wrap in TensorDataset / DataLoader.
      7. Return (train_loader, test_loader, EmberNet(...), (mean, std),
         Mode.MultiClassClassification or .MultiLabelClassification).

    For testing the pipeline end-to-end without EMBER, use
    `create_synthetic_tabular_datasets` instead.
    """
    raise NotImplementedError(
        "EMBER2024 loading not implemented yet. Use "
        "create_synthetic_tabular_datasets for pipeline validation, or "
        "implement following the TODOs in this docstring."
    )
