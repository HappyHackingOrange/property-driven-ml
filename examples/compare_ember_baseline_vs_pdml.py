"""
Head-to-head: standard cross-entropy training vs property-driven training,
both on the same EMBER2024 slice, both evaluated against the same attack.
Multi-seed replication: each seed varies the data subsample, the model
initialization, and the DataLoader shuffle order; results are reported
as mean ± std across seeds.

The constraint is ``NonFunctionalRobustnessConstraint``: an attacker may
perturb only the EMBER feature groups that correspond to "non-functional"
file attributes (byte histogram, byte-entropy distribution, string
statistics, section metadata, rich header, PE-parsing warnings). The
remaining groups (general, header, imports, exports, datadirectories,
authenticode) are clamped to their original values.

What the script prints:

  - per-seed: prediction accuracy, constraint sat (random), constraint
    security (PGD), for both baseline and property-driven models.
  - aggregate: mean ± std across all seeds.

Run:

    PYTHONPATH=. uv run python -m examples.compare_ember_baseline_vs_pdml
"""

import statistics
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import property_driven_ml.logics as logics
import property_driven_ml.training as training
from property_driven_ml.training import train

from examples.datasets.ember import create_ember_datasets
from examples.malware_constraints import NonFunctionalRobustnessConstraint


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_baseline(model, device, loader, epochs):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        stamp(
            f"  baseline epoch {epoch + 1}: "
            f"loss={total_loss / n:.3f} acc={correct / n:.3f}"
        )


def train_property_driven(
    model, device, loader, constraint, oracle, logic, epochs, mode
):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        info = train(
            epoch=epoch,
            N=model,
            device=device,
            train_loader=loader,
            optimizer=optimizer,
            oracle=oracle,
            logic=logic,
            constraint=constraint,
            with_dl=True,
            mode=mode,
            alpha=0.5,
        )
        stamp(
            f"  pdml epoch {epoch}: "
            f"acc={info.pred_metric:.3f} "
            f"constr_sec={info.constr_sec:.3f}"
        )


def evaluate(model, device, test_loader, constraint, logic, attack_logic, mean, std):
    """Return (pred_acc, constr_acc_random, constr_sec_adversarial) on test."""
    oracle = training.PGD(
        attack_logic, device, steps=20, restarts=4, step_size=0.01, mean=mean, std=std
    )
    model.eval()
    n_total = 0
    n_correct = 0
    n_constr_random = 0
    n_constr_secure = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            n_correct += (logits.argmax(1) == y).sum().item()
        # Random-sample constraint satisfaction: lo/hi from precondition, pick a random point.
        with torch.no_grad():
            _, sat_random = constraint.eval(
                model, x, None, y, logics.BooleanLogic(), reduction="sum"
            )
        # Adversarial: PGD inside the precondition.
        adv = oracle.attack(model, x, y, constraint)
        with torch.no_grad():
            _, sat_adv = constraint.eval(
                model, x, adv, y, logics.BooleanLogic(), reduction="sum"
            )
        n_total += len(y)
        n_constr_random += sat_random.item()
        n_constr_secure += sat_adv.item()
    return (
        n_correct / n_total,
        n_constr_random / n_total,
        n_constr_secure / n_total,
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
    """Train one baseline + one property-driven model under the given seed
    and return per-metric results on the test split."""
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

    # Match initial weights between the two models so the only difference
    # is the training objective, not the init.
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
    return {
        "baseline_acc": base_acc,
        "baseline_constr_random": base_cr,
        "baseline_constr_sec": base_cs,
        "pdml_acc": pdml_acc,
        "pdml_constr_random": pdml_cr,
        "pdml_constr_sec": pdml_cs,
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
            f"  seed {seed} done in {time.time() - t0:.1f}s | "
            f"baseline: acc={r['baseline_acc']:.3f} sec={r['baseline_constr_sec']:.3f} | "
            f"pdml: acc={r['pdml_acc']:.3f} sec={r['pdml_constr_sec']:.3f}"
        )
        results.append(r)

    print()
    print(f"=== Aggregate across {len(seeds)} seeds ===")
    print("  metric                          baseline (mean ± std)   pdml (mean ± std)")
    print(
        "  ----------------------------    ----------------------  ----------------------"
    )
    for key, label in [
        ("acc", "prediction accuracy"),
        ("constr_random", "constraint sat (random pert.)"),
        ("constr_sec", "constraint security (PGD)"),
    ]:
        b_vals = [r[f"baseline_{key}"] for r in results]
        p_vals = [r[f"pdml_{key}"] for r in results]
        print(f"  {label:30s}  {_fmt(b_vals):22s}  {_fmt(p_vals):22s}")
    print()


if __name__ == "__main__":
    main()
