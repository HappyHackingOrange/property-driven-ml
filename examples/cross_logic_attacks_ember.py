"""
Cross-logic attack transfer on EMBER2024: does a property-driven model
trained against logic A also defend against PGD attacks generated using
logic B?

If property-driven defenses are logic-specific, the defense would only
hold against attacks computed under the training logic. If the defense
generalizes, training with any "good" logic produces a model that
resists attacks generated under any other logic.

Protocol per seed:
  - Train one baseline (cross-entropy only).
  - For each training logic in {LeakyLogic, QLL, STL} -- the three logics
    that achieved full constraint security in sweep_logic_ember.py --
    train a separate property-driven model from the same initial weights.
  - Evaluate every model against PGD attacks using each of 5 attack
    logics, including 2 "weak" attack logics (DL2, GoedelFuzzy) to
    check whether the weak gradient signal still produces effective
    adversarials despite not defending well during training.

Output: rows are training logic, columns are attack logic, entries are
constraint security under that train/attack pair, averaged over seeds.

Run:

    PYTHONPATH=. uv run python -m examples.cross_logic_attacks_ember
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


# Training: only the 3 logics that achieved full security previously.
TRAIN_LOGIC_FACTORIES = [
    ("LeakyLogic", lambda: logics.LeakyLogic()),
    ("QLL", lambda: logics.QLL()),
    ("STL", lambda: logics.STL()),
]

# Attack: the 3 working + 2 representative "weak" logics to see whether
# defense transfers across gradient styles.
ATTACK_LOGIC_FACTORIES = [
    ("LeakyLogic", lambda: logics.LeakyLogic()),
    ("QLL", lambda: logics.QLL()),
    ("STL", lambda: logics.STL()),
    ("DL2", lambda: logics.DL2()),
    ("GoedelFuzzy", lambda: logics.GoedelFuzzyLogic()),
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
    """Train baseline + 3 PDML models, then evaluate all of them under
    each of the 5 attack logics. Returns nested dict
    {model_name: {attack_logic: security}}.
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

    # Train each PDML model in turn.
    pdml_models: dict[str, torch.nn.Module] = {}
    for train_name, train_factory in TRAIN_LOGIC_FACTORIES:
        stamp(f"  seed {seed}, training PDML with {train_name}")
        _, _, m, _, _ = create_ember_datasets(
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
        )
        m = m.to(device)
        m.load_state_dict(init_state)
        train_logic = train_factory()
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
        train_property_driven(
            m,
            device,
            train_loader,
            constraint_train,
            oracle_train,
            train_logic,
            epochs,
            mode,
        )
        pdml_models[train_name] = m

    # Now sweep attack logics against every model (including baseline).
    results: dict[str, dict[str, float]] = {}
    constraint_eval = NonFunctionalRobustnessConstraint(
        device=device,
        epsilon=epsilon,
        delta=delta,
        std=std,
    )

    all_models = {"__baseline__": model_baseline, **pdml_models}
    for model_name, model in all_models.items():
        results[model_name] = {}
        for attack_name, attack_factory in ATTACK_LOGIC_FACTORIES:
            attack_logic = attack_factory()
            _, _, sec = evaluate(
                model,
                device,
                test_loader,
                constraint_eval,
                attack_logic,
                attack_logic,
                mean,
                std,
            )
            results[model_name][attack_name] = sec
            stamp(
                f"  seed {seed}: model={model_name:12s} "
                f"attack={attack_name:14s}  sec={sec:.3f}"
            )

    return results


def _fmt(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    mean = statistics.mean(vals)
    if len(vals) > 1:
        return f"{mean:.3f}±{statistics.stdev(vals):.3f}"
    return f"{mean:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2]
    max_samples = 20_000
    epochs = 3
    epsilon = 0.5
    delta = 0.5

    # aggregate[model_name][attack_name] = list of per-seed values
    aggregate: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        per_seed = run_one_seed(
            seed=seed,
            device=device,
            max_samples=max_samples,
            epochs=epochs,
            epsilon=epsilon,
            delta=delta,
        )
        for model_name, attack_results in per_seed.items():
            for attack_name, sec in attack_results.items():
                aggregate.setdefault(model_name, {}).setdefault(attack_name, []).append(
                    sec
                )
        stamp(f"  seed {seed} done in {time.time() - t0:.1f}s")

    print()
    print("=== Constraint security: rows = training logic, cols = attack logic ===")
    print(f"=== {len(seeds)} seeds, {max_samples} samples, {epochs} epochs ===")
    # Header
    col_width = 14
    attack_names = [name for name, _ in ATTACK_LOGIC_FACTORIES]
    header = "  " + "trained ↓ / attack →".ljust(20)
    for attack_name in attack_names:
        header += f"  {attack_name:<{col_width}s}"
    print(header)
    print("  " + "-" * (20 + (col_width + 2) * len(attack_names)))

    def row(model_name: str, label: str):
        line = f"  {label:20s}"
        for attack_name in attack_names:
            vals = aggregate.get(model_name, {}).get(attack_name, [])
            line += f"  {_fmt(vals):<{col_width}s}"
        print(line)

    row("__baseline__", "(baseline, CE only)")
    for train_name, _ in TRAIN_LOGIC_FACTORIES:
        row(train_name, train_name)
    print()


if __name__ == "__main__":
    main()
