"""
Logic sweep on EMBER2024: does the 0 to 1 PGD-security jump observed
with LeakyLogic generalize across the framework's differentiable logics?
Addresses RQ3 in the DDSA proposal: are current differentiable logics
sufficient for adversarial classification constraints?

Protocol per seed:
  - Train one baseline (cross-entropy only).
  - For each logic in the sweep, train a separate property-driven model
    from the same initial weights using that logic in both the
    constraint-loss path and the inner PGD attack used during training.
  - Evaluate every property-driven model with a *common* attack
    (PGD-QLL) so security numbers are comparable across logics.

The 7 logics tested span the three main families in the framework:

  - DL2-style penalty logics: DL2, LeakyLogic, QLL.
  - Classical fuzzy logics: Goedel, Lukasiewicz, Reichenbach.
  - Signal temporal logic: STL.

(Yager omitted because of upstream issue #11; the remaining fuzzy
variants -- KleeneDienes, Goguen, ReichenbachSigmoidal, RealProductLogic
-- are excluded just to keep the sweep bounded at 7 logics x 3 seeds.)

Run:

    PYTHONPATH=. uv run python -m examples.sweep_logic_ember
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


LOGIC_FACTORIES = [
    ("LeakyLogic", lambda: logics.LeakyLogic()),
    ("DL2", lambda: logics.DL2()),
    ("QLL", lambda: logics.QLL()),
    ("GoedelFuzzy", lambda: logics.GoedelFuzzyLogic()),
    ("LukasiewiczFuzzy", lambda: logics.LukasiewiczFuzzyLogic()),
    ("ReichenbachFuzzy", lambda: logics.ReichenbachFuzzyLogic()),
    ("STL", lambda: logics.STL()),
]


def run_one_seed(
    seed: int,
    device: torch.device,
    max_samples: int,
    epochs: int,
    epsilon: float,
    delta: float,
    batch_size: int = 256,
) -> dict:
    """Train baseline once, then a property-driven model per logic, all
    from the same initial weights. Evaluate all with a common QLL attack.
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
    init_state = {k: v.clone() for k, v in model_baseline.state_dict().items()}

    stamp(f"  seed {seed}: training baseline")
    train_baseline(model_baseline, device, train_loader, epochs)

    common_attack_logic = logics.QLL()  # comparable attack across all PDML variants
    results: dict[str, dict] = {}

    # First record baseline metrics (independent of training logic).
    constraint_eval = NonFunctionalRobustnessConstraint(
        device=device,
        epsilon=epsilon,
        delta=delta,
        std=std,
    )
    base_acc, base_cr, base_cs = evaluate(
        model_baseline,
        device,
        test_loader,
        constraint_eval,
        common_attack_logic,
        common_attack_logic,
        mean,
        std,
    )
    results["__baseline__"] = {
        "acc": base_acc,
        "constr_random": base_cr,
        "constr_sec": base_cs,
    }
    stamp(f"  seed {seed}, baseline: acc={base_acc:.3f} sec={base_cs:.3f}")

    for logic_name, logic_factory in LOGIC_FACTORIES:
        stamp(f"  seed {seed}, logic={logic_name}: training pdml")
        _, _, model_pdml, _, _ = create_ember_datasets(
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
        )
        model_pdml = model_pdml.to(device)
        model_pdml.load_state_dict(init_state)

        train_logic = logic_factory()
        constraint_train = NonFunctionalRobustnessConstraint(
            device=device,
            epsilon=epsilon,
            delta=delta,
            std=std,
        )
        oracle_train = training.PGD(
            train_logic,
            device,
            steps=10,
            restarts=2,
            step_size=0.01,
            mean=mean,
            std=std,
        )
        try:
            train_property_driven(
                model_pdml,
                device,
                train_loader,
                constraint_train,
                oracle_train,
                train_logic,
                epochs,
                mode,
            )
            p_acc, p_cr, p_cs = evaluate(
                model_pdml,
                device,
                test_loader,
                constraint_eval,
                common_attack_logic,
                common_attack_logic,
                mean,
                std,
            )
            results[logic_name] = {
                "acc": p_acc,
                "constr_random": p_cr,
                "constr_sec": p_cs,
            }
            stamp(f"  seed {seed}, logic={logic_name}: acc={p_acc:.3f} sec={p_cs:.3f}")
        except Exception as e:
            stamp(
                f"  seed {seed}, logic={logic_name}: FAILED with {type(e).__name__}: {e}"
            )
            results[logic_name] = {
                "acc": None,
                "constr_random": None,
                "constr_sec": None,
            }

    return results


def _fmt(vals: list) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "FAILED"
    mean = statistics.mean(vals)
    if len(vals) > 1:
        return f"{mean:.3f} ± {statistics.stdev(vals):.3f}"
    return f"{mean:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2]
    max_samples = 20_000
    epochs = 3
    epsilon = 0.5
    delta = 0.5

    # Aggregate results per logic
    by_logic: dict[str, list[dict]] = {}

    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        per_logic = run_one_seed(
            seed=seed,
            device=device,
            max_samples=max_samples,
            epochs=epochs,
            epsilon=epsilon,
            delta=delta,
        )
        for logic_name, metrics in per_logic.items():
            by_logic.setdefault(logic_name, []).append(metrics)
        stamp(f"  seed {seed} done in {time.time() - t0:.1f}s")

    print()
    print(f"=== Aggregate across {len(seeds)} seeds ===")
    print("  logic                accuracy            constraint security (PGD-QLL)")
    print("  -----------------    ----------------    -----------------------------")
    # Baseline first
    if "__baseline__" in by_logic:
        rs = by_logic["__baseline__"]
        acc = _fmt([r["acc"] for r in rs])
        sec = _fmt([r["constr_sec"] for r in rs])
        print(f"  {'(baseline, CE only)':17s}    {acc:16s}    {sec:16s}")
    for logic_name, _ in LOGIC_FACTORIES:
        if logic_name not in by_logic:
            continue
        rs = by_logic[logic_name]
        acc = _fmt([r["acc"] for r in rs])
        sec = _fmt([r["constr_sec"] for r in rs])
        print(f"  {logic_name:17s}    {acc:16s}    {sec:16s}")
    print()


if __name__ == "__main__":
    main()
