"""
Per-class-cap sweep for the family-consistency constraint. The headline
test of experiment 6's prediction: does the constraint's value grow as the
family-classification task becomes data-starved?

Caps every family at C train samples and sweeps C. The test set and class
set are fixed across caps (gathered/selected once), so only train size
varies. Reports baseline vs constrained on overall accuracy, rare-family
macro-F1, and constraint satisfaction.

Run:

    PYTHONPATH=. uv run python -m examples.sweep_family_cap_ember
"""

import statistics
import time

import torch

from examples.datasets.ember_family import (
    gather_win32_family_pool,
    select_classes,
    build_loaders_from_pool,
)
from examples.compare_ember_family import run_one_seed


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt(vals):
    m = statistics.mean(vals)
    return f"{m:.3f}±{statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{m:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    caps = [25, 50, 100, 200, 500]
    epochs = 5
    margin = 1.0

    pool, behav, coverage = gather_win32_family_pool(weeks=2)
    commons, rares = select_classes(pool, coverage)

    rows = []
    for cap in caps:
        loaders = build_loaders_from_pool(
            pool, behav, commons, rares, per_class_cap=cap, split_seed=0
        )
        ntr = sum(len(y) for _, y in loaders["train_loader"])
        base, cons = [], []
        for seed in seeds:
            b, c = run_one_seed(seed, device, loaders, margin, epochs)
            base.append(b)
            cons.append(c)
        rows.append((cap, ntr, base, cons))
        stamp(
            f"cap={cap} (train={ntr}): "
            f"baseline rareF1={statistics.mean([r['rare_f1'] for r in base]):.3f} "
            f"sat={statistics.mean([r['constr_sat'] for r in base]):.3f} | "
            f"cons rareF1={statistics.mean([r['rare_f1'] for r in cons]):.3f} "
            f"sat={statistics.mean([r['constr_sat'] for r in cons]):.3f}"
        )

    print()
    print(
        f"=== Family-consistency vs per-class cap ({len(seeds)} seeds, {epochs} epochs, margin={margin}) ==="
    )
    print("  cap   train   metric        baseline          constrained")
    print("  ---   -----   -----------   ---------------   ---------------")
    for cap, ntr, base, cons in rows:
        for key, label in [
            ("rare_f1", "rare macro-F1"),
            ("acc", "overall acc"),
            ("constr_sat", "constr sat"),
        ]:
            b = _fmt([r[key] for r in base])
            c = _fmt([r[key] for r in cons])
            tag = f"{cap:5d} {ntr:6d}" if key == "rare_f1" else "          "
            print(f"  {tag}   {label:11s}   {b:15s}   {c:15s}")
        print()


if __name__ == "__main__":
    main()
