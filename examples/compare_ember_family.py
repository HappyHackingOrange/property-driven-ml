"""
Head-to-head: standard cross-entropy training vs family-consistency
training on the EMBER2024 Win32 family-classification task. Seventh
experiment, third of the four DDSA constraint families, and the first on
the multiclass family task.

The constraint (FamilyConsistencyConstraint): for a sample whose true
family is f, no behaviorally-incompatible family g may score within a
margin of f. Incompatibility is derived from the dataset's own behavior
tags (cosine of behavior-tag profiles < tau); see ember_family.py.

Hypothesis (from experiment 6's low-data result): the constraint should
help most for rare families, where standard training has too few samples
to learn family identity, while incompatible common families dominate.

Metrics: overall accuracy, macro-F1 (the long tail makes accuracy alone
misleading), constraint satisfaction, and accuracy/macro-F1 split by
common vs rare families.

Run:

    PYTHONPATH=. uv run python -m examples.compare_ember_family
"""

import statistics
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

import property_driven_ml.logics as logics
from examples.models import EmberNet
from examples.datasets.ember_family import (
    gather_win32_family_pool,
    select_classes,
    build_loaders_from_pool,
)
from examples.malware_constraints import (
    FamilyConsistencyConstraint,
    IdentityAttack,
    EMBER_FEATURE_DIM,
)
from examples.compare_ember_baseline_vs_pdml import (
    train_baseline,
    train_property_driven,
)


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_family(model, device, loader, constraint, n_classes, common_idx, rare_idx):
    model.eval()
    preds, trues = [], []
    sat = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            trues.append(y.cpu())
            _, s = constraint.eval(
                model, x, x, y, logics.BooleanLogic(), reduction="sum"
            )
        sat += s.item()
        n += len(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    labels = list(range(n_classes))
    acc = float((preds == trues).mean())
    macro_f1 = f1_score(trues, preds, labels=labels, average="macro", zero_division=0)
    common = sorted(common_idx)
    rare = sorted(rare_idx)
    common_mask = np.isin(trues, common)
    rare_mask = np.isin(trues, rare)
    common_acc = float((preds[common_mask] == trues[common_mask]).mean())
    rare_acc = (
        float((preds[rare_mask] == trues[rare_mask]).mean()) if rare_mask.any() else 0.0
    )
    common_f1 = f1_score(trues, preds, labels=common, average="macro", zero_division=0)
    rare_f1 = f1_score(trues, preds, labels=rare, average="macro", zero_division=0)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "constr_sat": sat / n,
        "common_acc": common_acc,
        "rare_acc": rare_acc,
        "common_f1": common_f1,
        "rare_f1": rare_f1,
    }


def run_one_seed(seed, device, loaders, margin, epochs):
    train_loader = loaders["train_loader"]
    test_loader = loaders["test_loader"]
    incompatible_index = loaders["incompatible_index"]
    n_classes = len(loaders["class_names"])
    common_idx = loaders["common_idx"]
    rare_idx = loaders["rare_idx"]
    mode = loaders["mode"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    init = EmberNet(input_dim=EMBER_FEATURE_DIM, n_classes=n_classes).state_dict()

    model_b = EmberNet(input_dim=EMBER_FEATURE_DIM, n_classes=n_classes).to(device)
    model_b.load_state_dict(init)
    train_baseline(model_b, device, train_loader, epochs)

    model_c = EmberNet(input_dim=EMBER_FEATURE_DIM, n_classes=n_classes).to(device)
    model_c.load_state_dict(init)
    logic = logics.LeakyLogic()
    constraint = FamilyConsistencyConstraint(device, incompatible_index, margin=margin)
    oracle = IdentityAttack(logic, device)
    train_property_driven(
        model_c, device, train_loader, constraint, oracle, logic, epochs, mode
    )

    eval_constraint = FamilyConsistencyConstraint(
        device, incompatible_index, margin=margin
    )
    b = evaluate_family(
        model_b, device, test_loader, eval_constraint, n_classes, common_idx, rare_idx
    )
    c = evaluate_family(
        model_c, device, test_loader, eval_constraint, n_classes, common_idx, rare_idx
    )
    return b, c


def _fmt(vals):
    m = statistics.mean(vals)
    return f"{m:.3f} ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else f"{m:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    epochs = 5
    margin = 1.0
    per_class_cap = 500

    # Gather and select classes once; data fixed across seeds (split_seed=0).
    pool, behav, coverage = gather_win32_family_pool(weeks=2)
    commons, rares = select_classes(pool, coverage)
    loaders = build_loaders_from_pool(
        pool, behav, commons, rares, per_class_cap=per_class_cap, split_seed=0
    )
    ntr = sum(len(y) for _, y in loaders["train_loader"])
    nte = sum(len(y) for _, y in loaders["test_loader"])
    n_inc = sum(len(s) for s in loaders["incompatible_index"]) // 2
    stamp(
        f"task: {len(loaders['class_names'])} classes "
        f"({len(commons)} common + {len(rares)} rare), "
        f"train={ntr} test={nte}, incompatible pairs={n_inc}, margin={margin}"
    )

    base, cons = [], []
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        b, c = run_one_seed(seed, device, loaders, margin, epochs)
        stamp(
            f"  seed {seed} ({time.time() - t0:.1f}s)  "
            f"baseline: acc={b['acc']:.3f} rareF1={b['rare_f1']:.3f} sat={b['constr_sat']:.3f} | "
            f"cons: acc={c['acc']:.3f} rareF1={c['rare_f1']:.3f} sat={c['constr_sat']:.3f}"
        )
        base.append(b)
        cons.append(c)

    print()
    print(
        f"=== Aggregate across {len(seeds)} seeds (cap={per_class_cap}, margin={margin}, {epochs} epochs) ==="
    )
    print("  metric                      baseline            family-consistency")
    print("  ------------------------    ----------------    ----------------")
    for key, label in [
        ("acc", "overall accuracy"),
        ("macro_f1", "macro-F1 (all)"),
        ("common_f1", "macro-F1 (common)"),
        ("rare_f1", "macro-F1 (rare)"),
        ("common_acc", "accuracy (common)"),
        ("rare_acc", "accuracy (rare)"),
        ("constr_sat", "constraint satisfaction"),
    ]:
        b = _fmt([r[key] for r in base])
        c = _fmt([r[key] for r in cons])
        print(f"  {label:24s}    {b:16s}    {c:16s}")
    print()


if __name__ == "__main__":
    main()
