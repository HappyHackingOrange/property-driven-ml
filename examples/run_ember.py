"""
End-to-end smoke test of the property-driven-ml pipeline on real EMBER2024
data, exercising the same code path as ``run_synthetic_tabular.py`` but
against the actual malware-classification benchmark.

Loads a subsampled slice (defaults to 50k training rows, 10k test rows)
so the smoke test stays fast on a workstation. Drop ``max_samples`` for
the full ~5M-row training run.

Prerequisites:

  1. Install thrember (``uv sync --group ember`` or
     ``pip install -e ".[ember]"``).
  2. Download and vectorize the dataset once:
         import thrember
         thrember.download_dataset("~/data/ember2024")
         thrember.create_vectorized_features("~/data/ember2024")
     (Takes ~30 min download + ~20 min vectorize at typical bandwidth.)

Run:

    PYTHONPATH=. uv run python -m examples.run_ember
"""

import time

import torch
import torch.optim as optim

import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml.training import train, test

from examples.datasets.ember import create_ember_datasets


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp(f"device: {device}")

    batch_size = 256
    epochs = 3
    max_samples = 50_000  # train rows; test gets max_samples // 5
    epsilon = 0.05
    delta = 0.5

    stamp(f"Loading EMBER2024 (max_samples={max_samples})...")
    t0 = time.time()
    train_loader, test_loader, model, (mean, std), mode = create_ember_datasets(
        batch_size=batch_size,
        max_samples=max_samples,
    )
    stamp(f"  loader built in {time.time() - t0:.1f}s")
    stamp(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    model = model.to(device)

    logic = logics.LeakyLogic()
    constraint = constraints.StrongClassificationRobustnessConstraint(
        device=device,
        epsilon=epsilon,
        delta=delta,
        std=std,
    )

    oracle_self_train = training.PGD(
        logic, device, steps=10, restarts=2, step_size=0.01, mean=mean, std=std
    )
    oracle_self_test = training.PGD(
        logic, device, steps=20, restarts=4, step_size=0.01, mean=mean, std=std
    )
    oracle_common_test = training.PGD(
        logics.QLL(), device, steps=20, restarts=4, step_size=0.01, mean=mean, std=std
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_info = train(
            epoch=epoch,
            N=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            oracle=oracle_self_train,
            logic=logic,
            constraint=constraint,
            with_dl=True,
            mode=mode,
            alpha=0.5,
        )
        stamp(
            f"epoch {epoch} TRAIN  ({time.time() - t0:.1f}s)  "
            f"pred_acc={train_info.pred_metric:.3f}  "
            f"constr_sec={train_info.constr_sec:.3f}  "
            f"pred_loss={train_info.pred_loss:.3f}  "
            f"constr_loss={train_info.constr_loss:.3f}"
        )

        t0 = time.time()
        test_info = test(
            epoch=epoch,
            N=model,
            device=device,
            test_loader=test_loader,
            oracle_self=oracle_self_test,
            oracle_common=oracle_common_test,
            logic=logic,
            constraint=constraint,
            is_baseline=False,
            mode=mode,
        )
        stamp(
            f"epoch {epoch}  TEST  ({time.time() - t0:.1f}s)  "
            f"pred_acc={test_info.pred_metric:.3f}  "
            f"constr_acc={test_info.constr_acc:.3f}  "
            f"constr_sec_self={test_info.constr_sec_self:.3f}  "
            f"constr_sec_common={test_info.constr_sec_common:.3f}"
        )


if __name__ == "__main__":
    main()
