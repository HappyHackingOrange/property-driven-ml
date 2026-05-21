"""
Epsilon sweep: how does the masked-robustness constraint trade off
prediction accuracy and constraint security as the threat-model strength
varies? Produces the security-vs-epsilon curve that's the canonical
figure in an adversarial-robustness paper.

Protocol per seed:
  - Train one baseline (cross-entropy only) on the seed's data slice.
  - For each epsilon in the sweep, train a separate property-driven model
    from the same initial weights using that epsilon in the constraint.
  - Evaluate both models at every epsilon (the PGD attacker's ball
    matches the eval epsilon, regardless of what the model was trained
    on).

Output is a table: rows are epsilon values, columns are (baseline_acc,
baseline_sec, pdml_acc, pdml_sec), each entry mean ± std across seeds.

Run:

    PYTHONPATH=. uv run python -m examples.sweep_epsilon_ember
"""

import statistics
import time

import numpy as np
import torch

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


def run_one_seed(
    seed: int,
    device: torch.device,
    max_samples: int,
    epochs: int,
    epsilons: list[float],
    delta: float,
    batch_size: int = 256,
) -> dict:
    """Train one baseline and one property-driven model per epsilon under
    a given seed. Returns nested dict ``{eps: {metric: value}}`` with
    metrics for both models at each eval epsilon.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader, model_baseline, (mean, std), mode = (
        create_ember_datasets(
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
        )
    )
    model_baseline = model_baseline.to(device)

    # Snapshot the initial weights so each property-driven model trains
    # from the same starting point as the baseline.
    init_state = {k: v.clone() for k, v in model_baseline.state_dict().items()}

    stamp(f"  seed {seed}: training baseline")
    train_baseline(model_baseline, device, train_loader, epochs)

    results = {}
    logic_eval = logics.QLL()
    for eps in epsilons:
        stamp(f"  seed {seed}, eps={eps}: training pdml")
        _, _, model_pdml, _, _ = create_ember_datasets(
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
        )
        model_pdml = model_pdml.to(device)
        model_pdml.load_state_dict(init_state)

        constraint_train = NonFunctionalRobustnessConstraint(
            device=device,
            epsilon=eps,
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
            epsilon=eps,
            delta=delta,
            std=std,
        )
        base_acc, base_cr, base_cs = evaluate(
            model_baseline,
            device,
            test_loader,
            constraint_eval,
            logic_eval,
            logic_eval,
            mean,
            std,
        )
        pdml_acc, pdml_cr, pdml_cs = evaluate(
            model_pdml,
            device,
            test_loader,
            constraint_eval,
            logic_eval,
            logic_eval,
            mean,
            std,
        )
        results[eps] = {
            "baseline_acc": base_acc,
            "baseline_constr_sec": base_cs,
            "pdml_acc": pdml_acc,
            "pdml_constr_sec": pdml_cs,
        }
        stamp(
            f"  seed {seed}, eps={eps}: "
            f"baseline acc={base_acc:.3f} sec={base_cs:.3f} | "
            f"pdml acc={pdml_acc:.3f} sec={pdml_cs:.3f}"
        )
    return results


def _fmt(vals: list[float]) -> str:
    mean = statistics.mean(vals)
    if len(vals) > 1:
        return f"{mean:.3f}±{statistics.stdev(vals):.3f}"
    return f"{mean:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [
        0,
        1,
        2,
    ]  # 3 seeds (vs 5 for the single-eps comparison) to keep wall-time bounded
    max_samples = 20_000
    epochs = 3
    epsilons = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    delta = 0.5

    # results[eps] = list of per-seed result dicts
    all_results: dict[float, list[dict]] = {eps: [] for eps in epsilons}

    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        per_eps = run_one_seed(
            seed=seed,
            device=device,
            max_samples=max_samples,
            epochs=epochs,
            epsilons=epsilons,
            delta=delta,
        )
        for eps, metrics in per_eps.items():
            all_results[eps].append(metrics)
        stamp(f"  seed {seed} done in {time.time() - t0:.1f}s")

    print()
    print(
        f"=== Aggregate across {len(seeds)} seeds, {max_samples} samples, {epochs} epochs ==="
    )
    print("  eps       baseline_acc     baseline_sec     pdml_acc         pdml_sec")
    print(
        "  -------   -------------    -------------    -------------    -------------"
    )
    for eps in epsilons:
        rs = all_results[eps]
        base_acc = _fmt([r["baseline_acc"] for r in rs])
        base_sec = _fmt([r["baseline_constr_sec"] for r in rs])
        pdml_acc = _fmt([r["pdml_acc"] for r in rs])
        pdml_sec = _fmt([r["pdml_constr_sec"] for r in rs])
        print(
            f"  {eps:7.2f}   {base_acc:14s}   {base_sec:14s}   {pdml_acc:14s}   {pdml_sec:14s}"
        )
    print()


if __name__ == "__main__":
    main()
