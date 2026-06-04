"""
Temporal feature-stability measurement for EMBER2024 (experiment 8).

Splits the 52-week training window into early (weeks 0-25) and late (weeks
26-51) halves and computes, per feature, how much its predictive
relationship with the label shifts between halves (label-association shift)
and how much its distribution shifts (distributional shift). The
temporally-unstable features are the tail of the label-association shift;
the temporal-invariance constraint forces the model to be invariant to
them.

Per-sample time is recoverable from week_id (0-51). Format mix is balanced
across halves by equal per-file sampling, so the measured shift reflects
temporal drift, not a format-mix artifact.

The computed shift arrays are cached next to the dataset so the experiment
scripts share one measurement.
"""

import glob
import json
import os
import time

import numpy as np

from thrember import PEFeatureExtractor

TRAIN_GLOB = "/home/shadowcipher/data/ember2024/*_train.jsonl"
CACHE_PATH = "/home/shadowcipher/data/ember2024/temporal_stability.npz"
EMBER_FEATURE_DIM = 2568


def _week_of(path):
    return os.path.basename(path)[:21]


def compute_stability(per_file: int = 60, verbose: bool = True):
    """Gather an early-half and late-half mixed sample (balanced per file
    across all weeks and formats), z-score globally, and return
    (assoc_shift, dist_shift) arrays of length EMBER_FEATURE_DIM.

    assoc_shift[i] = |corr(feature_i, label)_early - corr(feature_i, label)_late|
    dist_shift[i]  = |mean_early - mean_late| in global-std units
    """
    ext = PEFeatureExtractor()
    files = sorted(glob.glob(TRAIN_GLOB))
    weeks = sorted({_week_of(f) for f in files})
    early_weeks, late_weeks = set(weeks[:26]), set(weeks[26:])

    def gather(file_list):
        X, y = [], []
        for f in file_list:
            n = 0
            with open(f) as fh:
                for line in fh:
                    if n >= per_file:
                        break
                    r = json.loads(line)
                    lab = r.get("label")
                    if lab not in (0, 1):
                        continue
                    X.append(ext.process_raw_features(r).astype(np.float32))
                    y.append(lab)
                    n += 1
        return np.stack(X), np.array(y, dtype=np.float64)

    t0 = time.time()
    Xe, ye = gather([f for f in files if _week_of(f) in early_weeks])
    Xl, yl = gather([f for f in files if _week_of(f) in late_weeks])

    allX = np.concatenate([Xe, Xl])
    mu = allX.mean(0)
    sd = allX.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Ez, Lz = (Xe - mu) / sd, (Xl - mu) / sd
    dist_shift = np.abs(Ez.mean(0) - Lz.mean(0))

    def corr(Z, yv):
        num = ((Z - Z.mean(0)) * (yv - yv.mean())[:, None]).mean(0)
        den = Z.std(0) * yv.std() + 1e-9
        return num / den

    assoc_shift = np.abs(corr(Ez, ye) - corr(Lz, yl))
    if verbose:
        print(
            f"temporal stability: early={len(Xe)} late={len(Xl)} "
            f"in {time.time() - t0:.0f}s"
        )
    return assoc_shift.astype(np.float64), dist_shift.astype(np.float64)


def get_stability(use_cache: bool = True):
    """Return (assoc_shift, dist_shift), computing and caching if needed."""
    if use_cache and os.path.isfile(CACHE_PATH):
        d = np.load(CACHE_PATH)
        return d["assoc_shift"], d["dist_shift"]
    assoc_shift, dist_shift = compute_stability()
    try:
        np.savez(CACHE_PATH, assoc_shift=assoc_shift, dist_shift=dist_shift)
    except OSError:
        pass
    return assoc_shift, dist_shift


def unstable_indices(assoc_shift, top_frac: float = 0.15) -> np.ndarray:
    """Indices of the top `top_frac` most temporally-unstable features by
    label-association shift."""
    thr = np.quantile(assoc_shift, 1.0 - top_frac)
    return np.where(assoc_shift >= thr)[0]


if __name__ == "__main__":
    assoc, dist = get_stability(use_cache=False)
    for frac in (0.05, 0.10, 0.15, 0.20):
        idx = unstable_indices(assoc, frac)
        print(f"top {int(frac * 100):2d}%: {len(idx)} unstable features")
