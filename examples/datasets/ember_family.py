"""
EMBER2024 Win32 malware-family dataset for the family-consistency
experiment (experiment 7).

Family classification is a Win32/PE concept and only meaningful for
malicious samples (benign files have family=None). This module streams the
Win32 train jsonl files, vectorizes each record on the fly via thrember's
PEFeatureExtractor (no dependence on the binary X_train.dat), selects a
tractable class set (common families plus a deliberate rare tail), and
derives a family-incompatibility matrix from the dataset's own behavior
tags.

Nothing here encodes hand-supplied malware-family domain facts: the class
set, the behavior profiles, and the incompatibility matrix are all derived
from the EMBER2024 labels.
"""

import glob
import json
import time
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from thrember import PEFeatureExtractor

from property_driven_ml.training.mode import Mode
from examples.models import EmberNet


WIN32_TRAIN_GLOB = "/home/shadowcipher/data/ember2024/*Win32_train.jsonl"
WIN32_TEST_GLOB = "/home/shadowcipher/data/ember2024/*Win32_test.jsonl"
EMBER_FEATURE_DIM = 2568


def gather_win32_family_pool(
    weeks: int = 2,
    per_family_cap: int = 1200,
    split_glob: str = WIN32_TRAIN_GLOB,
    verbose: bool = True,
):
    """Stream the first `weeks` Win32 jsonl files; for malicious family-labeled
    rows, vectorize on the fly and collect per-family feature vectors (capped)
    plus per-family behavior-tag counts.

    Returns (pool, behavior_profiles, behavior_coverage):
      pool[family] -> list of float32 feature vectors (length <= per_family_cap)
      behavior_profiles[family] -> Counter of behavior tags
      behavior_coverage[family] -> total behavior-tag occurrences seen
    """
    ext = PEFeatureExtractor()
    files = sorted(glob.glob(split_glob))[:weeks]
    pool = defaultdict(list)
    behav = defaultdict(Counter)
    t0 = time.time()
    for f in files:
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("label") != 1:
                    continue
                fam = rec.get("family")
                if not fam:
                    continue
                # behavior counts are gathered from ALL family rows (uncapped)
                # so profiles are accurate even once the feature cap is hit.
                for b in rec.get("behavior") or []:
                    behav[fam][b] += 1
                if len(pool[fam]) < per_family_cap:
                    pool[fam].append(ext.process_raw_features(rec).astype(np.float32))
    coverage = {fam: sum(c.values()) for fam, c in behav.items()}
    if verbose:
        print(
            f"gathered {sum(len(v) for v in pool.values())} vectors "
            f"across {len(pool)} families from {len(files)} Win32 weeks "
            f"in {time.time() - t0:.0f}s"
        )
    return pool, behav, coverage


def select_classes(
    pool,
    coverage,
    n_common: int = 12,
    n_rare: int = 8,
    common_min: int = 400,
    rare_range=(30, 100),
    min_behavior_coverage: int = 50,
    seed: int = 0,
):
    """Pick `n_common` common families (>= common_min gathered, adequate
    behavior coverage) and `n_rare` rare families (gathered count within
    rare_range, adequate behavior coverage). Returns (common, rare)."""
    counts = {fam: len(v) for fam, v in pool.items()}
    eligible = {fam for fam in pool if coverage.get(fam, 0) >= min_behavior_coverage}
    commons = sorted(
        (f for f in eligible if counts[f] >= common_min),
        key=lambda f: counts[f],
        reverse=True,
    )[:n_common]
    rare_pool = sorted(
        f for f in eligible if rare_range[0] <= counts[f] <= rare_range[1]
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(rare_pool)
    rares = sorted(rare_pool[:n_rare])
    return commons, rares


def dominant_behavior(behav_counter):
    tot = sum(behav_counter.values())
    if tot == 0:
        return None, 0.0
    b, c = behav_counter.most_common(1)[0]
    return b, c / tot


# "packed" is a file property (EMBER carries a separate "packer" field), not
# a malicious behavior; it co-occurs across unrelated families and inflates
# spurious profile overlap, so it is excluded from incompatibility profiles.
IGNORE_BEHAVIOR_TAGS = {"packed"}


def _profile_vector(behav_counter, tag_index):
    v = np.zeros(len(tag_index), dtype=np.float64)
    for b, c in behav_counter.items():
        if b in tag_index:
            v[tag_index[b]] = c
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_incompatibility(behav, classes, tau: float = 0.2):
    """Family g is incompatible with family f if the cosine similarity of
    their behavior-tag profiles is below tau. Returns a dict
    family -> set(incompatible families) restricted to `classes`."""
    all_tags = sorted(
        {t for f in classes for t in behav[f] if t not in IGNORE_BEHAVIOR_TAGS}
    )
    tag_index = {t: i for i, t in enumerate(all_tags)}
    profiles = {f: _profile_vector(behav[f], tag_index) for f in classes}
    incompat = {f: set() for f in classes}
    for i, f in enumerate(classes):
        for g in classes:
            if f == g:
                continue
            sim = float(np.dot(profiles[f], profiles[g]))
            if sim < tau:
                incompat[f].add(g)
    return incompat


def build_loaders_from_pool(
    pool,
    behav,
    commons,
    rares,
    batch_size: int = 256,
    per_class_cap: Optional[int] = 500,
    test_per_class: int = 40,
    tau: float = 0.2,
    split_seed: int = 0,
):
    """Build family-classification loaders from a pre-gathered pool and a
    fixed class selection. The test set (test_per_class per class) is held
    out before applying per_class_cap, so a cap sweep varies only the train
    set with the test set and class definitions held constant.

    Returns a dict: train_loader, test_loader, model, class_names,
    common_idx, rare_idx, incompatible_index (list[set[int]]), mode, and
    derived metadata.
    """
    classes = list(commons) + list(rares)
    incompat = build_incompatibility(behav, classes, tau=tau)
    class_to_idx = {f: i for i, f in enumerate(classes)}
    incompatible_index = [{class_to_idx[g] for g in incompat[f]} for f in classes]

    rng = np.random.default_rng(split_seed)
    Xtr, ytr, Xte, yte = [], [], [], []
    for f in classes:
        vecs = pool[f]
        idx = rng.permutation(len(vecs))
        n_test = min(test_per_class, len(vecs) // 3)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        if per_class_cap is not None:
            train_idx = train_idx[:per_class_cap]
        for j in train_idx:
            Xtr.append(vecs[j])
            ytr.append(class_to_idx[f])
        for j in test_idx:
            Xte.append(vecs[j])
            yte.append(class_to_idx[f])

    Xtr = np.stack(Xtr).astype(np.float32)
    Xte = np.stack(Xte).astype(np.float32)
    ytr = np.array(ytr, dtype=np.int64)
    yte = np.array(yte, dtype=np.int64)

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=batch_size,
        shuffle=False,
    )
    model = EmberNet(input_dim=EMBER_FEATURE_DIM, n_classes=len(classes))

    common_idx = {class_to_idx[f] for f in commons}
    rare_idx = {class_to_idx[f] for f in rares}

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "model": model,
        "class_names": classes,
        "common_idx": common_idx,
        "rare_idx": rare_idx,
        "incompatible_index": incompatible_index,
        "behav": behav,
        "pool_counts": {f: len(pool[f]) for f in classes},
        "mode": Mode.MultiClassClassification,
        "mean": mean,
        "std": std,
    }


def create_family_datasets(
    batch_size: int = 256,
    weeks: int = 2,
    n_common: int = 12,
    n_rare: int = 8,
    per_class_cap: Optional[int] = 500,
    test_per_class: int = 40,
    tau: float = 0.2,
    seed: int = 0,
):
    """Convenience: gather the pool, select classes, and build loaders in one
    call. Experiments that sweep per_class_cap should instead gather and
    select once, then call build_loaders_from_pool per cap so the class set
    and test split stay fixed across the sweep."""
    pool, behav, coverage = gather_win32_family_pool(
        weeks=weeks, per_family_cap=max(per_class_cap or 1200, 1200) + test_per_class
    )
    commons, rares = select_classes(
        pool, coverage, n_common=n_common, n_rare=n_rare, seed=seed
    )
    return build_loaders_from_pool(
        pool,
        behav,
        commons,
        rares,
        batch_size=batch_size,
        per_class_cap=per_class_cap,
        test_per_class=test_per_class,
        tau=tau,
        split_seed=seed,
    )


if __name__ == "__main__":
    # Print the derived taxonomy for hand-verification before any modeling.
    pool, behav, coverage = gather_win32_family_pool(weeks=2)
    commons, rares = select_classes(pool, coverage)
    classes = commons + rares
    print()
    print("COMMON families (count, behavior_coverage, dominant behavior):")
    for f in commons:
        b, share = dominant_behavior(behav[f])
        print(f"  {f:16s} n={len(pool[f]):5d} cov={coverage[f]:6d}  {b}:{share:.2f}")
    print("RARE families:")
    for f in rares:
        b, share = dominant_behavior(behav[f])
        print(f"  {f:16s} n={len(pool[f]):5d} cov={coverage[f]:6d}  {b}:{share:.2f}")
    incompat = build_incompatibility(behav, classes, tau=0.2)
    n_pairs = sum(len(s) for s in incompat.values()) // 2
    total_pairs = len(classes) * (len(classes) - 1) // 2
    print()
    print(f"incompatible (cosine<0.2) pairs: {n_pairs} of {total_pairs} possible")
    print("per-family incompatible counts:")
    for f in classes:
        print(f"  {f:16s} incompatible with {len(incompat[f])} families")
