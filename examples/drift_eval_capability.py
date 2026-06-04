"""
Concept-drift evaluation of the capability-monotonicity constraint on
EMBER2024. Stretch hypothesis from the experiment plan: capability-based
reasoning might be more temporally stable than surface statistics, so the
capability-constrained model might show a smaller accuracy-drift gap than
the baseline.

Protocol (mirrors drift_eval_ember.py):
  - Hold out a random 20% of the train split as an in-distribution
    validation set (same 52-week period as train).
  - Train baseline (CE) and capability-monotonicity model on the other 80%
    from the same initial weights.
  - Evaluate accuracy on (a) the in-dist holdout and (b) EMBER's natural
    12-week-later test split. drift_gap = acc(in_dist) - acc(temporal).
  - Also report monotonicity satisfaction on both splits.

Run:

    PYTHONPATH=. uv run python -m examples.drift_eval_capability
"""

import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import property_driven_ml.logics as logics

from examples.datasets.ember import create_ember_datasets
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


def split_train_indist(train_loader, val_fraction, batch_size, seed):
    full = train_loader.dataset
    n = len(full)
    n_val = int(n * val_fraction)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return (
        DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False),
    )


def run_one_seed(seed, device, max_samples, epochs, delta, margin, batch_size=256):
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_train_loader, temporal_test_loader, model_baseline, (mean, std), mode = (
        create_ember_datasets(batch_size=batch_size, max_samples=max_samples, seed=seed)
    )
    model_baseline = model_baseline.to(device)
    train_loader, indist_loader = split_train_indist(
        full_train_loader, 0.2, batch_size, seed
    )

    _, _, model_cap, _, _ = create_ember_datasets(
        batch_size=batch_size, max_samples=max_samples, seed=seed
    )
    model_cap = model_cap.to(device)
    model_cap.load_state_dict(model_baseline.state_dict())

    train_baseline(model_baseline, device, train_loader, epochs)

    logic = logics.LeakyLogic()
    constraint = CapabilityMonotonicityConstraint(
        device=device, delta=delta, margin=margin
    )
    oracle = CapabilityRaiseAttack(logic, device, mean=mean, std=std)
    train_property_driven(
        model_cap, device, train_loader, constraint, oracle, logic, epochs, mode
    )

    ec = CapabilityMonotonicityConstraint(device=device, delta=delta, margin=margin)
    b_in_acc, b_in_sat = evaluate_monotonicity(
        model_baseline, device, indist_loader, ec
    )
    b_tp_acc, b_tp_sat = evaluate_monotonicity(
        model_baseline, device, temporal_test_loader, ec
    )
    c_in_acc, c_in_sat = evaluate_monotonicity(model_cap, device, indist_loader, ec)
    c_tp_acc, c_tp_sat = evaluate_monotonicity(
        model_cap, device, temporal_test_loader, ec
    )
    return {
        "baseline_indist_acc": b_in_acc,
        "baseline_temporal_acc": b_tp_acc,
        "baseline_drift_gap": b_in_acc - b_tp_acc,
        "baseline_indist_sat": b_in_sat,
        "baseline_temporal_sat": b_tp_sat,
        "cap_indist_acc": c_in_acc,
        "cap_temporal_acc": c_tp_acc,
        "cap_drift_gap": c_in_acc - c_tp_acc,
        "cap_indist_sat": c_in_sat,
        "cap_temporal_sat": c_tp_sat,
    }


def _fmt(vals):
    m = statistics.mean(vals)
    if len(vals) > 1:
        return f"{m:.3f} ± {statistics.stdev(vals):.3f}"
    return f"{m:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    max_samples = 20_000
    epochs = 3
    delta = 1.0
    margin = 0.0

    results = []
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        r = run_one_seed(seed, device, max_samples, epochs, delta, margin)
        stamp(
            f"  seed {seed} done in {time.time() - t0:.1f}s\n"
            f"    baseline: in={r['baseline_indist_acc']:.3f} temp={r['baseline_temporal_acc']:.3f} gap={r['baseline_drift_gap']:+.3f}\n"
            f"    cap:      in={r['cap_indist_acc']:.3f} temp={r['cap_temporal_acc']:.3f} gap={r['cap_drift_gap']:+.3f}"
        )
        results.append(r)

    print()
    print(f"=== Aggregate across {len(seeds)} seeds (delta={delta}) ===")
    print("  metric                          baseline                capability-mono")
    print(
        "  ----------------------------    --------------------    --------------------"
    )
    for key, label in [
        ("indist_acc", "in-distribution accuracy"),
        ("temporal_acc", "temporal accuracy"),
        ("drift_gap", "accuracy drift gap"),
        ("indist_sat", "in-dist monotonicity sat"),
        ("temporal_sat", "temporal monotonicity sat"),
    ]:
        b = _fmt([r[f"baseline_{key}"] for r in results])
        c = _fmt([r[f"cap_{key}"] for r in results])
        print(f"  {label:30s}  {b:20s}  {c:20s}")
    print()


if __name__ == "__main__":
    main()
