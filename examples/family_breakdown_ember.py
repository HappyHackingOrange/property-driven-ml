"""
Per-family breakdown of the family-consistency constraint (experiment 7,
deepened). Tests the reviewer hypothesis (Bruni, June 10 meeting) that the
constraint's averaged effect, "noticeable but small," hides a larger effect
concentrated in specific data-scarce families.

Same task and protocol as compare_ember_family (20-class Win32, cap=500,
margin=1.0, 5 epochs, identical init per seed), but instead of aggregate
metrics it reports, PER FAMILY and ranked by improvement:

  - train samples for that family (the natural-scarcity axis: commons are
    capped at 500, rares land at their gathered count minus the test split)
  - per-family F1 and recall, baseline vs constrained, averaged over seeds
  - per-family constraint satisfaction, baseline vs constrained
  - the F1 delta (constrained minus baseline) and its correlation with
    log train count across families

Run:

    PYTHONPATH=. uv run python -m examples.family_breakdown_ember
"""

import json
import statistics
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

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


def per_family_eval(model, device, loader, constraint, n_classes):
    """Per-class F1, recall, and constraint satisfaction on the test set."""
    model.eval()
    preds, trues, sats = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            trues.append(y.cpu())
            _, sat = constraint.eval(
                model, x, x, y, logics.BooleanLogic(), reduction=None
            )
        sats.append(sat.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    sats = torch.cat(sats).numpy()
    labels = list(range(n_classes))
    f1 = f1_score(trues, preds, labels=labels, average=None, zero_division=0)
    rec = recall_score(trues, preds, labels=labels, average=None, zero_division=0)
    sat_per_class = np.array(
        [sats[trues == c].mean() if (trues == c).any() else 0.0 for c in labels]
    )
    return f1, rec, sat_per_class


def run_one_seed(seed, device, loaders, margin, epochs):
    train_loader = loaders["train_loader"]
    test_loader = loaders["test_loader"]
    incompatible_index = loaders["incompatible_index"]
    n_classes = len(loaders["class_names"])
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

    ev = FamilyConsistencyConstraint(device, incompatible_index, margin=margin)
    b = per_family_eval(model_b, device, test_loader, ev, n_classes)
    c = per_family_eval(model_c, device, test_loader, ev, n_classes)
    return b, c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    seeds = [0, 1, 2, 3, 4]
    epochs = 5
    margin = 1.0
    per_class_cap = 500

    pool, behav, coverage = gather_win32_family_pool(weeks=2)
    commons, rares = select_classes(pool, coverage)
    loaders = build_loaders_from_pool(
        pool, behav, commons, rares, per_class_cap=per_class_cap, split_seed=0
    )
    classes = loaders["class_names"]
    n_classes = len(classes)
    rare_idx = loaders["rare_idx"]

    # Train-sample count per class (after test holdout and cap), the
    # natural-scarcity axis.
    train_counts = np.zeros(n_classes, dtype=int)
    for _, y in loaders["train_loader"]:
        for c in y.numpy():
            train_counts[c] += 1
    stamp(
        f"task: {n_classes} classes, train counts min={train_counts.min()} "
        f"max={train_counts.max()}"
    )

    b_f1s, b_recs, b_sats = [], [], []
    c_f1s, c_recs, c_sats = [], [], []
    for seed in seeds:
        stamp(f"=== seed {seed} ===")
        t0 = time.time()
        (bf, br, bs), (cf, cr, cs) = run_one_seed(
            seed, device, loaders, margin, epochs
        )
        stamp(f"  seed {seed} done in {time.time() - t0:.0f}s")
        b_f1s.append(bf)
        b_recs.append(br)
        b_sats.append(bs)
        c_f1s.append(cf)
        c_recs.append(cr)
        c_sats.append(cs)

    b_f1 = np.mean(b_f1s, axis=0)
    c_f1 = np.mean(c_f1s, axis=0)
    b_rec = np.mean(b_recs, axis=0)
    c_rec = np.mean(c_recs, axis=0)
    b_sat = np.mean(b_sats, axis=0)
    c_sat = np.mean(c_sats, axis=0)
    d_f1 = c_f1 - b_f1
    d_sat = c_sat - b_sat

    order = np.argsort(-d_f1)
    print()
    print(
        f"=== Per-family breakdown ({len(seeds)} seeds, cap={per_class_cap}, "
        f"margin={margin}, {epochs} epochs), ranked by F1 delta ==="
    )
    print(
        "  family             n_train  rare   base_F1  cons_F1  dF1      "
        "base_sat  cons_sat  dSat"
    )
    for i in order:
        tag = "RARE" if i in rare_idx else "    "
        print(
            f"  {classes[i]:16s}  {train_counts[i]:6d}  {tag}   "
            f"{b_f1[i]:.3f}    {c_f1[i]:.3f}    {d_f1[i]:+.3f}   "
            f"{b_sat[i]:.3f}     {c_sat[i]:.3f}     {d_sat[i]:+.3f}"
        )

    # Scarcity correlation: does the F1 improvement concentrate at low n?
    logn = np.log(train_counts.astype(float))
    corr_f1 = float(np.corrcoef(logn, d_f1)[0, 1])
    corr_sat = float(np.corrcoef(logn, d_sat)[0, 1])
    rare_mask = np.array([i in rare_idx for i in range(n_classes)])
    print()
    print(f"  corr(log n_train, F1 delta)  = {corr_f1:+.3f}")
    print(f"  corr(log n_train, sat delta) = {corr_sat:+.3f}")
    print(
        f"  mean F1 delta: rare={d_f1[rare_mask].mean():+.4f}  "
        f"common={d_f1[~rare_mask].mean():+.4f}"
    )
    print(
        f"  mean sat delta: rare={d_sat[rare_mask].mean():+.4f}  "
        f"common={d_sat[~rare_mask].mean():+.4f}"
    )
    print()
    print(
        "JSON_RESULT "
        + json.dumps(
            {
                "classes": list(classes),
                "train_counts": train_counts.tolist(),
                "rare": rare_mask.tolist(),
                "base_f1": b_f1.tolist(),
                "cons_f1": c_f1.tolist(),
                "base_rec": b_rec.tolist(),
                "cons_rec": c_rec.tolist(),
                "base_sat": b_sat.tolist(),
                "cons_sat": c_sat.tolist(),
                "corr_f1": corr_f1,
                "corr_sat": corr_sat,
            }
        )
    )


if __name__ == "__main__":
    main()
