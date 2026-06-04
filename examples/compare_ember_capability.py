"""
Head-to-head: standard cross-entropy training vs capability-monotonicity
training, on the same EMBER2024 slice. Sixth experiment in the EMBER
series, and the first POSITIVE domain-rule constraint (the prior five all
concern robustness, a negative property).

The constraint (CapabilityMonotonicityConstraint, see malware_constraints.py):
adding the process-injection capability to any file must not lower its
malicious score by more than a margin. Realized by raising the 12
process-injection import-hash buckets and checking
malicious_score(raised) >= malicious_score(original) - margin.

Why monotonicity rather than "injection => malicious": Step-1 measurement
on EMBER2024 Win32 showed process injection (capa ATT&CK T1055) is only
~54% malicious at ~0.7% prevalence, essentially uncorrelated and rare. No
capa capability was a strong malicious indicator. A hard implication rule
would be false ~46% of the time. Monotonicity only claims the capability
is non-negative evidence (never evidence toward benign), which IS
directionally supported, and it is a global property needing no rare
subset.

Metrics reported per model:
  - prediction accuracy on the test split
  - monotonicity satisfaction: fraction of test samples where raising the
    injection capability does not drop the malicious score beyond margin

Run:

    PYTHONPATH=. uv run python -m examples.compare_ember_capability
"""

import statistics
import time

import numpy as np
import torch

import property_driven_ml.logics as logics
from property_driven_ml.training.mode import Mode  # noqa: F401  (documents the task mode)

from examples.datasets.ember import create_ember_datasets
from examples.malware_constraints import (
    CapabilityMonotonicityConstraint,
    CapabilityRaiseAttack,
)
from examples.compare_ember_baseline_vs_pdml import (
    train_baseline,
    train_property_driven,
)


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_monotonicity(model, device, loader, constraint):
    """Return (pred_acc, monotonicity_sat) on the loader. monotonicity_sat
    is the fraction of samples where raising the injection capability does
    not drop the malicious score beyond the constraint margin."""
    model.eval()
    n = 0
    correct = 0
    sat = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            x_raised = constraint.raised_corner(x)
            _, s = constraint.eval(
                model, x, x_raised, y, logics.BooleanLogic(), reduction="sum"
            )
        n += len(y)
        sat += s.item()
    return correct / n, sat / n


def run_one_seed(
    seed: int,
    device: torch.device,
    max_samples: int,
    epochs: int,
    delta: float,
    margin: float,
    batch_size: int = 256,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader, model_baseline, (mean, std), mode = (
        create_ember_datasets(batch_size=batch_size, max_samples=max_samples, seed=seed)
    )
    model_baseline = model_baseline.to(device)

    # Matched-init capability model.
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

    eval_constraint = CapabilityMonotonicityConstraint(
        device=device, delta=delta, margin=margin
    )
    b_acc, b_sat = evaluate_monotonicity(
        model_baseline, device, test_loader, eval_constraint
    )
    c_acc, c_sat = evaluate_monotonicity(
        model_cap, device, test_loader, eval_constraint
    )
    return {
        "baseline_acc": b_acc,
        "baseline_mono_sat": b_sat,
        "cap_acc": c_acc,
        "cap_mono_sat": c_sat,
    }


def _fmt(vals: list[float]) -> str:
    mean = statistics.mean(vals)
    if len(vals) > 1:
        return f"{mean:.3f} ± {statistics.stdev(vals):.3f}"
    return f"{mean:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    max_samples = 20_000
    epochs = 3
    delta = 1.0  # raise injection buckets by 1 normalized unit
    margin = 0.0  # strict monotonicity

    results = []
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        r = run_one_seed(
            seed=seed,
            device=device,
            max_samples=max_samples,
            epochs=epochs,
            delta=delta,
            margin=margin,
        )
        stamp(
            f"  seed {seed} done in {time.time() - t0:.1f}s | "
            f"baseline: acc={r['baseline_acc']:.3f} mono_sat={r['baseline_mono_sat']:.3f} | "
            f"cap: acc={r['cap_acc']:.3f} mono_sat={r['cap_mono_sat']:.3f}"
        )
        results.append(r)

    print()
    print(
        f"=== Aggregate across {len(seeds)} seeds (delta={delta}, margin={margin}) ==="
    )
    print("  metric                          baseline                capability-mono")
    print(
        "  ----------------------------    --------------------    --------------------"
    )
    for key, label in [
        ("acc", "prediction accuracy"),
        ("mono_sat", "monotonicity satisfaction"),
    ]:
        b = _fmt([r[f"baseline_{key}"] for r in results])
        c = _fmt([r[f"cap_{key}"] for r in results])
        print(f"  {label:30s}  {b:20s}  {c:20s}")
    print()


if __name__ == "__main__":
    main()
