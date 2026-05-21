"""
Concept-drift evaluation on EMBER2024: does property-driven training with
the non-functional-robustness constraint reduce the accuracy drop on
future malware?

EMBER2024 ships a natural temporal split: 52 weeks of training data,
then 12 weeks of test data. To turn that into a drift measurement, we:

  - Hold out a random 20% slice of the *train* split as an
    in-distribution validation set (same 52-week period as train).
  - Train on the remaining 80%.
  - Evaluate on (a) the in-dist holdout and (b) EMBER's natural test
    split (the 12 weeks after train).
  - drift_gap = accuracy(in_dist) - accuracy(temporal).

If property-driven training makes the model rely more on features that
are stable across time (frozen groups: imports, header, exports, ...)
and less on features that drift (non-functional groups: byte
distributions, section names, ...), the gap should shrink for the
property-driven model relative to the baseline.

This addresses RQ2 in the DDSA proposal.

Run:

    PYTHONPATH=. uv run python -m examples.drift_eval_ember
"""

import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import property_driven_ml.logics as logics
import property_driven_ml.training as training

from examples.datasets.ember import create_ember_datasets
from examples.malware_constraints import NonFunctionalRobustnessConstraint
from examples.compare_ember_baseline_vs_pdml import (
    train_baseline,
    train_property_driven,
    evaluate,
)


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_train_indist(
    train_loader: DataLoader, val_fraction: float, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Split a train DataLoader's underlying dataset 80/20 into a new
    train loader and an in-distribution validation loader. The val
    fraction is taken uniformly at random from the same temporal period
    as the train data."""
    full = train_loader.dataset
    n = len(full)
    n_val = int(n * val_fraction)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    val_ds = Subset(full, val_idx)
    train_ds = Subset(full, train_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def run_one_seed(
    seed: int,
    device: torch.device,
    max_samples: int,
    epochs: int,
    epsilon: float,
    delta: float,
    batch_size: int = 256,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_train_loader, temporal_test_loader, model_baseline, (mean, std), mode = (
        create_ember_datasets(
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
        )
    )
    model_baseline = model_baseline.to(device)

    # Split train into train_real (80%) and in-distribution val (20%).
    train_loader, indist_val_loader = split_train_indist(
        full_train_loader,
        val_fraction=0.2,
        batch_size=batch_size,
        seed=seed,
    )

    # Matched-init property-driven model.
    _, _, model_pdml, _, _ = create_ember_datasets(
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
    )
    model_pdml = model_pdml.to(device)
    model_pdml.load_state_dict(model_baseline.state_dict())

    train_baseline(model_baseline, device, train_loader, epochs)

    constraint_train = NonFunctionalRobustnessConstraint(
        device=device,
        epsilon=epsilon,
        delta=delta,
        std=std,
    )
    logic_train = logics.LeakyLogic()
    oracle_train = training.PGD(
        logic_train,
        device,
        steps=10,
        restarts=2,
        step_size=0.01,
        mean=mean,
        std=std,
    )
    train_property_driven(
        model_pdml,
        device,
        train_loader,
        constraint_train,
        oracle_train,
        logic_train,
        epochs,
        mode,
    )

    constraint_eval = NonFunctionalRobustnessConstraint(
        device=device,
        epsilon=epsilon,
        delta=delta,
        std=std,
    )
    logic_eval = logics.QLL()

    b_in_acc, _, b_in_sec = evaluate(
        model_baseline,
        device,
        indist_val_loader,
        constraint_eval,
        logic_eval,
        logic_eval,
        mean,
        std,
    )
    b_tp_acc, _, b_tp_sec = evaluate(
        model_baseline,
        device,
        temporal_test_loader,
        constraint_eval,
        logic_eval,
        logic_eval,
        mean,
        std,
    )
    p_in_acc, _, p_in_sec = evaluate(
        model_pdml,
        device,
        indist_val_loader,
        constraint_eval,
        logic_eval,
        logic_eval,
        mean,
        std,
    )
    p_tp_acc, _, p_tp_sec = evaluate(
        model_pdml,
        device,
        temporal_test_loader,
        constraint_eval,
        logic_eval,
        logic_eval,
        mean,
        std,
    )
    return {
        "baseline_indist_acc": b_in_acc,
        "baseline_temporal_acc": b_tp_acc,
        "baseline_drift_gap": b_in_acc - b_tp_acc,
        "baseline_indist_sec": b_in_sec,
        "baseline_temporal_sec": b_tp_sec,
        "pdml_indist_acc": p_in_acc,
        "pdml_temporal_acc": p_tp_acc,
        "pdml_drift_gap": p_in_acc - p_tp_acc,
        "pdml_indist_sec": p_in_sec,
        "pdml_temporal_sec": p_tp_sec,
    }


def _fmt(vals: list[float]) -> str:
    mean = statistics.mean(vals)
    if len(vals) > 1:
        return (
            f"{mean:+.3f} ± {statistics.stdev(vals):.3f}"
            if any(v < 0 for v in vals)
            else f"{mean:.3f} ± {statistics.stdev(vals):.3f}"
        )
    return f"{mean:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    max_samples = 20_000
    epochs = 3
    epsilon = 0.5
    delta = 0.5

    results = []
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        r = run_one_seed(
            seed=seed,
            device=device,
            max_samples=max_samples,
            epochs=epochs,
            epsilon=epsilon,
            delta=delta,
        )
        stamp(
            f"  seed {seed} done in {time.time() - t0:.1f}s\n"
            f"    baseline: in_dist_acc={r['baseline_indist_acc']:.3f} "
            f"temporal_acc={r['baseline_temporal_acc']:.3f} "
            f"drift_gap={r['baseline_drift_gap']:+.3f}\n"
            f"    pdml:     in_dist_acc={r['pdml_indist_acc']:.3f} "
            f"temporal_acc={r['pdml_temporal_acc']:.3f} "
            f"drift_gap={r['pdml_drift_gap']:+.3f}"
        )
        results.append(r)

    print()
    print(f"=== Aggregate across {len(seeds)} seeds ===")
    print("  metric                                baseline                pdml")
    print(
        "  ----------------------------------    --------------------    --------------------"
    )
    for key, label in [
        ("indist_acc", "in-distribution accuracy"),
        ("temporal_acc", "temporal accuracy"),
        ("drift_gap", "drift gap (indist - temporal)"),
        ("indist_sec", "in-distribution constr security"),
        ("temporal_sec", "temporal constr security"),
    ]:
        b = _fmt([r[f"baseline_{key}"] for r in results])
        p = _fmt([r[f"pdml_{key}"] for r in results])
        print(f"  {label:34s}    {b:20s}    {p:20s}")
    print()


if __name__ == "__main__":
    main()
