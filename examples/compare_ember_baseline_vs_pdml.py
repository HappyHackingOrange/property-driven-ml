"""
Head-to-head: standard cross-entropy training vs property-driven training,
both on the same EMBER2024 slice, both evaluated against the same attack.

The constraint is ``NonFunctionalRobustnessConstraint``: an attacker may
perturb only the EMBER feature groups that correspond to "non-functional"
file attributes (byte histogram, byte-entropy distribution, string
statistics, section metadata, rich header, PE-parsing warnings). The
remaining groups (general, header, imports, exports, datadirectories,
authenticode) are clamped to their original values.

What the script prints:

  - prediction accuracy on the test split (both models)
  - constraint satisfaction on random samples inside the precondition
    (both models)
  - constraint security under PGD adversarial attack with the masked
    epsilon ball (both models)

Run:

    PYTHONPATH=. uv run python -m examples.compare_ember_baseline_vs_pdml
"""

import time

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    max_samples = 20_000
    epochs = 3
    epsilon = 0.5
    delta = 0.5

    stamp(f"Loading EMBER2024 (max_samples={max_samples})...")
    train_loader, test_loader, model_baseline, (mean, std), mode = (
        create_ember_datasets(
            batch_size=256,
            max_samples=max_samples,
        )
    )
    model_baseline = model_baseline.to(device)

    # Get a fresh copy of the same architecture for the property-driven model
    # so the two runs are matched.
    _, _, model_pdml, _, _ = create_ember_datasets(
        batch_size=256,
        max_samples=max_samples,
    )
    model_pdml = model_pdml.to(device)
    # Force the same initial weights so the comparison isn't init-noise.
    model_pdml.load_state_dict(model_baseline.state_dict())

    stamp("=== Training baseline (cross-entropy only) ===")
    t0 = time.time()
    train_baseline(model_baseline, device, train_loader, epochs)
    stamp(f"  baseline trained in {time.time() - t0:.1f}s")

    stamp("=== Training property-driven (cross-entropy + constraint loss) ===")
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
    t0 = time.time()
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
    stamp(f"  property-driven trained in {time.time() - t0:.1f}s")

    stamp("=== Evaluation on test split ===")
    constraint_eval = NonFunctionalRobustnessConstraint(
        device=device,
        epsilon=epsilon,
        delta=delta,
        std=std,
    )
    logic_eval = logics.QLL()  # common reference logic for attacks
    stamp("  evaluating baseline...")
    base_acc, base_constr_random, base_constr_sec = evaluate(
        model_baseline,
        device,
        test_loader,
        constraint_eval,
        logic_eval,
        logic_eval,
        mean,
        std,
    )
    stamp("  evaluating pdml...")
    pdml_acc, pdml_constr_random, pdml_constr_sec = evaluate(
        model_pdml,
        device,
        test_loader,
        constraint_eval,
        logic_eval,
        logic_eval,
        mean,
        std,
    )

    print()
    print("  metric                          baseline    pdml")
    print("  ----------------------------    --------  --------")
    print(f"  prediction accuracy             {base_acc:8.3f}  {pdml_acc:8.3f}")
    print(
        f"  constraint sat (random pert.)   {base_constr_random:8.3f}  {pdml_constr_random:8.3f}"
    )
    print(
        f"  constraint security (PGD)       {base_constr_sec:8.3f}  {pdml_constr_sec:8.3f}"
    )
    print()


if __name__ == "__main__":
    main()
