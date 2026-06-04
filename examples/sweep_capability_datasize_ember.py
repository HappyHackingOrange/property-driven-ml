"""
Data-size sweep for the capability-monotonicity constraint on EMBER2024.

Tests the hypothesis the head-to-head result generated: standard training
learns the capability-monotonicity prior slowly and unreliably (epoch-1
baseline monotonicity was 0.73 +/- 0.32), so the constraint's value should
be CONCENTRATED in the low-data regime, where the model has not yet learned
the prior from data. That regime is where rare malware families and fresh
threats live.

Protocol per seed:
  - Load a fixed evaluation set (test split, ~4000 samples) and a 20k train
    pool, once. Snapshot one random init.
  - For each train size in the sweep, subsample the train pool to that size,
    train a baseline (CE) and a capability-constrained model from the SAME
    snapshot init, and evaluate both on the FIXED test set.
  - Holding the test set and init fixed isolates the effect of train size.

Reports, per train size: baseline vs capability accuracy and monotonicity
satisfaction. The hypothesis predicts the baseline-vs-capability
monotonicity gap (and baseline variance) grows as train size shrinks.

Run:

    PYTHONPATH=. uv run python -m examples.sweep_capability_datasize_ember
"""

import copy
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import property_driven_ml.logics as logics
from property_driven_ml.training.mode import Mode

from examples.datasets.ember import create_ember_datasets
from examples.models import EmberNet
from examples.malware_constraints import (
    CapabilityMonotonicityConstraint,
    CapabilityRaiseAttack,
)
from examples.compare_ember_baseline_vs_pdml import (
    train_baseline,
    train_property_driven,
)
from examples.compare_ember_capability import evaluate_monotonicity


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one_seed(seed, device, sizes, epochs, delta, margin, batch_size=256):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Fixed 20k train pool + ~4k test set for this seed.
    full_train_loader, test_loader, model0, (mean, std), mode = create_ember_datasets(
        batch_size=batch_size, max_samples=20_000, seed=seed
    )
    assert mode is Mode.MultiClassClassification
    init_state = copy.deepcopy(model0.state_dict())
    n_features = model0.fc1.in_features
    train_pool = full_train_loader.dataset
    pool_n = len(train_pool)
    rng = np.random.default_rng(seed)

    per_size = {}
    for size in sizes:
        size = min(size, pool_n)
        idx = rng.choice(pool_n, size=size, replace=False).tolist()
        train_loader = DataLoader(
            Subset(train_pool, idx), batch_size=batch_size, shuffle=True
        )

        model_b = EmberNet(input_dim=n_features, n_classes=2).to(device)
        model_b.load_state_dict(init_state)
        train_baseline(model_b, device, train_loader, epochs)

        model_c = EmberNet(input_dim=n_features, n_classes=2).to(device)
        model_c.load_state_dict(init_state)
        logic = logics.LeakyLogic()
        constraint = CapabilityMonotonicityConstraint(
            device=device, delta=delta, margin=margin
        )
        oracle = CapabilityRaiseAttack(logic, device, mean=mean, std=std)
        train_property_driven(
            model_c, device, train_loader, constraint, oracle, logic, epochs, mode
        )

        ec = CapabilityMonotonicityConstraint(device=device, delta=delta, margin=margin)
        b_acc, b_sat = evaluate_monotonicity(model_b, device, test_loader, ec)
        c_acc, c_sat = evaluate_monotonicity(model_c, device, test_loader, ec)
        per_size[size] = {
            "baseline_acc": b_acc,
            "baseline_mono": b_sat,
            "cap_acc": c_acc,
            "cap_mono": c_sat,
        }
    return per_size


def _fmt(vals):
    m = statistics.mean(vals)
    return f"{m:.3f}±{statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{m:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    sizes = [1000, 2000, 5000, 10000, 20000]
    epochs = 3
    delta = 1.0
    margin = 0.0

    agg = {s: [] for s in sizes}
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        per_size = run_one_seed(seed, device, sizes, epochs, delta, margin)
        for s, r in per_size.items():
            agg[s].append(r)
        stamp(f"  seed {seed} done in {time.time() - t0:.1f}s")

    print()
    print(
        f"=== Capability constraint vs train size ({len(seeds)} seeds, {epochs} epochs, delta={delta}) ==="
    )
    print("  train_n   baseline_acc    baseline_mono   cap_acc         cap_mono")
    print("  -------   -------------   -------------   -------------   -------------")
    for s in sizes:
        rs = agg[s]
        ba = _fmt([r["baseline_acc"] for r in rs])
        bm = _fmt([r["baseline_mono"] for r in rs])
        ca = _fmt([r["cap_acc"] for r in rs])
        cm = _fmt([r["cap_mono"] for r in rs])
        print(f"  {s:7d}   {ba:13s}   {bm:13s}   {ca:13s}   {cm:13s}")
    print()


if __name__ == "__main__":
    main()
