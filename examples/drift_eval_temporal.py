"""
Temporal-invariance constraint on EMBER2024: the eighth experiment and the
fourth DDSA constraint family. Targets the only fully negative result so
far (experiment 3: robustness generalizes across the temporal split but the
accuracy drift gap is unchanged, baseline 0.014 vs robustness-PDML 0.019).

The temporal-invariance constraint forces the model to be invariant to the
temporally-unstable features (those whose label association shifts most
between the early and late halves of the 52-week train window; see
ember_temporal.py). The hope: anchoring to stable features shrinks the
accuracy drift gap on the future 12-week window.

Same 20k-scale mixed pipeline as experiments 1-5 and the same drift
protocol as experiment 3, so the drift-gap comparison is clean.

Primary metric: accuracy drift gap (in-distribution minus temporal),
compared against experiment 3. Runs three things:
  - head-to-head: baseline vs temporal-invariance (top-15% unstable mask)
  - attribution: temporal(386) vs non-functional(1034) vs temporal-only(56)
  - threshold sweep: top 5/10/15/20% unstable

Run:

    PYTHONPATH=. uv run python -m examples.drift_eval_temporal
"""

import statistics
import time

import numpy as np
import torch

import property_driven_ml.logics as logics
import property_driven_ml.training as training
from examples.models import EmberNet
from examples.datasets.ember import create_ember_datasets
from examples.datasets.ember_temporal import get_stability, unstable_indices
from examples.malware_constraints import (
    TemporalInvarianceConstraint,
    feature_indices_for_groups,
    NON_FUNCTIONAL_GROUPS,
    EMBER_FEATURE_DIM,
)
from examples.compare_ember_baseline_vs_pdml import (
    train_baseline,
    train_property_driven,
)
from examples.drift_eval_ember import split_train_indist


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def accuracy(model, device, loader):
    model.eval()
    cor = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            cor += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return cor / n


def run_one_seed(seed, device, masks, epochs, epsilon, delta, max_samples=20000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_train, temporal_test, model0, (mean, std), mode = create_ember_datasets(
        batch_size=256, max_samples=max_samples, seed=seed
    )
    init = {k: v.clone() for k, v in model0.state_dict().items()}
    train_loader, indist_loader = split_train_indist(full_train, 0.2, 256, seed)

    # Baseline (mask-independent), shared across all masks this seed.
    model_b = EmberNet(input_dim=EMBER_FEATURE_DIM, n_classes=2).to(device)
    model_b.load_state_dict(init)
    train_baseline(model_b, device, train_loader, epochs)
    b_in = accuracy(model_b, device, indist_loader)
    b_tp = accuracy(model_b, device, temporal_test)
    out = {"__baseline__": (b_in, b_tp, b_in - b_tp)}

    logic = logics.LeakyLogic()
    for name, idx in masks.items():
        model_c = EmberNet(input_dim=EMBER_FEATURE_DIM, n_classes=2).to(device)
        model_c.load_state_dict(init)
        constraint = TemporalInvarianceConstraint(
            device,
            torch.as_tensor(idx, dtype=torch.long),
            epsilon=epsilon,
            delta=delta,
            std=std,
        )
        oracle = training.PGD(
            logic, device, steps=10, restarts=2, step_size=0.01, mean=mean, std=std
        )
        train_property_driven(
            model_c, device, train_loader, constraint, oracle, logic, epochs, mode
        )
        c_in = accuracy(model_c, device, indist_loader)
        c_tp = accuracy(model_c, device, temporal_test)
        out[name] = (c_in, c_tp, c_in - c_tp)
    return out


def _fmt(vals):
    m = statistics.mean(vals)
    return f"{m:+.3f}±{statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{m:+.3f}"


def _fmt_acc(vals):
    m = statistics.mean(vals)
    return f"{m:.3f}±{statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{m:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    epochs = 3
    epsilon = 0.5
    delta = 0.5

    assoc, _ = get_stability()
    nonfunc = feature_indices_for_groups(NON_FUNCTIONAL_GROUPS).numpy()
    t15 = unstable_indices(assoc, 0.15)
    temporal_only = np.array(sorted(set(t15.tolist()) - set(nonfunc.tolist())))
    masks = {
        "temporal05": unstable_indices(assoc, 0.05),
        "temporal10": unstable_indices(assoc, 0.10),
        "temporal15": t15,
        "temporal20": unstable_indices(assoc, 0.20),
        "nonfunc": nonfunc,
        "temporal_only56": temporal_only,
    }
    stamp("mask sizes: " + ", ".join(f"{k}={len(v)}" for k, v in masks.items()))

    agg = {k: [] for k in ["__baseline__", *masks]}
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        r = run_one_seed(seed, device, masks, epochs, epsilon, delta)
        for k, v in r.items():
            agg[k].append(v)
        stamp(f"  seed {seed} done in {time.time() - t0:.0f}s")

    print()
    print(
        f"=== Temporal-invariance drift evaluation ({len(seeds)} seeds, {epochs} epochs) ==="
    )
    print("experiment-3 reference: baseline gap 0.014, robustness-PDML gap 0.019")
    print()
    print("  model              in-dist acc       temporal acc      drift gap")
    print("  ----------------   ---------------   ---------------   ---------------")

    def row(key, label):
        rows = agg[key]
        ina = _fmt_acc([r[0] for r in rows])
        tpa = _fmt_acc([r[1] for r in rows])
        gap = _fmt([r[2] for r in rows])
        print(f"  {label:16s}   {ina:15s}   {tpa:15s}   {gap:15s}")

    row("__baseline__", "baseline (CE)")
    row("temporal15", "temporal-15% (HH)")
    print("  --- attribution ---")
    row("temporal15", "temporal (386)")
    row("nonfunc", "non-functional (1034)")
    row("temporal_only56", "temporal-only (56)")
    print("  --- threshold sweep ---")
    for k in ("temporal05", "temporal10", "temporal15", "temporal20"):
        row(k, k)
    print()


if __name__ == "__main__":
    main()
