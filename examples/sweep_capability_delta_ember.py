"""
Delta sweep for the capability-monotonicity constraint on EMBER2024: how
does the raise magnitude (how much injection capability we add) affect
baseline monotonicity satisfaction and the constraint's guarantee? Analog
of sweep_epsilon_ember.py for the positive domain constraint.

Run:

    PYTHONPATH=. uv run python -m examples.sweep_capability_delta_ember
"""

import statistics
import time

import torch

from examples.compare_ember_capability import run_one_seed


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt(vals):
    m = statistics.mean(vals)
    return f"{m:.3f}±{statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{m:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")
    seeds = [0, 1, 2]
    deltas = [0.5, 1.0, 2.0, 5.0, 10.0]
    max_samples = 20_000
    epochs = 3

    print()
    print(
        f"=== Capability delta sweep ({len(seeds)} seeds, {max_samples} samples, {epochs} epochs) ==="
    )
    print("  delta   baseline_acc     baseline_mono    cap_acc          cap_mono")
    print("  -----   -------------    -------------    -------------    -------------")
    for delta in deltas:
        rs = [
            run_one_seed(
                seed=s,
                device=device,
                max_samples=max_samples,
                epochs=epochs,
                delta=delta,
                margin=0.0,
            )
            for s in seeds
        ]
        ba = _fmt([r["baseline_acc"] for r in rs])
        bm = _fmt([r["baseline_mono_sat"] for r in rs])
        ca = _fmt([r["cap_acc"] for r in rs])
        cm = _fmt([r["cap_mono_sat"] for r in rs])
        print(f"  {delta:5.1f}   {ba:13s}    {bm:13s}    {ca:13s}    {cm:13s}")
    print()


if __name__ == "__main__":
    main()
