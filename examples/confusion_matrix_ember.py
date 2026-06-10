"""
Confusion-matrix and false-negative analysis for the binary EMBER2024 task:
does property-driven training change the error PROFILE, not just the
aggregate accuracy?

Motivated by reviewer feedback (Bruni, June 10 meeting): false negatives,
malware classified as benign, are the costly error in this domain, and the
head-to-head accuracy numbers say nothing about how the errors split.

Protocol matches experiment 1 (compare_ember_baseline_vs_pdml): 20k mixed
samples, 3 epochs, batch 256, eps=0.5, delta=0.5, LeakyLogic, identical
init per seed. The only addition is the full confusion matrix on the clean
test split (label 1 = malicious = positive class):

  FN  = malware classified benign  (the miss; the costly error)
  FP  = benign flagged as malware  (the false alarm)
  FNR = FN / (FN + TP), FPR = FP / (FP + TN)

Run:

    PYTHONPATH=. uv run python -m examples.confusion_matrix_ember
"""

import json
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
)


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def confusion(model, device, loader):
    """Return (tn, fp, fn, tp) on a binary loader with label 1 = malicious."""
    model.eval()
    tn = fp = fn = tp = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            tp += int(((pred == 1) & (y == 1)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())
    return tn, fp, fn, tp


def rates(tn, fp, fn, tp):
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "acc": (tp + tn) / max(tn + fp + fn + tp, 1),
        "fnr": fn / max(fn + tp, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def run_one_seed(seed, device, max_samples, epochs, epsilon, delta):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, model_b, (mean, std), mode = create_ember_datasets(
        batch_size=256, max_samples=max_samples, seed=seed
    )
    model_b = model_b.to(device)
    init = {k: v.clone() for k, v in model_b.state_dict().items()}

    train_baseline(model_b, device, train_loader, epochs)

    model_p = type(model_b)(input_dim=2568, n_classes=2).to(device)
    model_p.load_state_dict(init)
    logic = logics.LeakyLogic()
    constraint = NonFunctionalRobustnessConstraint(
        device=device, epsilon=epsilon, delta=delta, std=std
    )
    oracle = training.PGD(
        logic, device, steps=10, restarts=2, step_size=0.01, mean=mean, std=std
    )
    train_property_driven(
        model_p, device, train_loader, constraint, oracle, logic, epochs, mode
    )

    return rates(*confusion(model_b, device, test_loader)), rates(
        *confusion(model_p, device, test_loader)
    )


def _fmt(vals):
    m = statistics.mean(vals)
    return f"{m:.4f} ± {statistics.stdev(vals):.4f}" if len(vals) > 1 else f"{m:.4f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    max_samples = 20_000
    epochs = 3
    epsilon = 0.5
    delta = 0.5

    base, pdml = [], []
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        b, p = run_one_seed(seed, device, max_samples, epochs, epsilon, delta)
        stamp(
            f"  seed {seed} ({time.time() - t0:.0f}s)  "
            f"baseline: fn={b['fn']} fp={b['fp']} fnr={b['fnr']:.4f} | "
            f"pdml: fn={p['fn']} fp={p['fp']} fnr={p['fnr']:.4f}"
        )
        base.append(b)
        pdml.append(p)

    print()
    print(f"=== Confusion-matrix aggregate ({len(seeds)} seeds, {max_samples} samples) ===")
    print("  metric             baseline               property-driven")
    print("  ---------------    -------------------    -------------------")
    for key, label in [
        ("acc", "accuracy"),
        ("fnr", "FN rate (miss)"),
        ("fpr", "FP rate (alarm)"),
        ("precision", "precision"),
        ("recall", "recall"),
    ]:
        print(f"  {label:15s}    {_fmt([r[key] for r in base]):19s}    {_fmt([r[key] for r in pdml]):19s}")
    counts_b = {k: statistics.mean([r[k] for r in base]) for k in ("tn", "fp", "fn", "tp")}
    counts_p = {k: statistics.mean([r[k] for r in pdml]) for k in ("tn", "fp", "fn", "tp")}
    print(f"  mean counts (baseline): {counts_b}")
    print(f"  mean counts (pdml):     {counts_p}")
    print()
    print("JSON_RESULT " + json.dumps({"baseline": base, "pdml": pdml}))


if __name__ == "__main__":
    main()
