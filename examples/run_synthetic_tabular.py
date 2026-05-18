"""
End-to-end smoke test of the property-driven-ml pipeline on a synthetic
tabular dataset, exercising the same code path that a real EMBER run
will eventually use.

The dataset is two Gaussian clusters in a 64-dim feature space (binary
classification). The constraint is StrongClassificationRobustness inside an
L-infinity epsilon-ball: under any small perturbation of the input, the
model should still predict the true label. Trains for a few epochs and
reports prediction accuracy alongside constraint satisfaction on
adversarial (PGD) samples.

This script is intentionally small and self-contained. It is the minimum
needed to convince yourself the framework works on tabular numerical data
before plugging in real EMBER2024 features.
"""

import torch
import torch.optim as optim

import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training
from property_driven_ml.training import train, test

from examples.datasets.synthetic_tabular import create_synthetic_tabular_datasets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    batch_size = 64
    epochs = 3
    epsilon = 0.05  # L-infinity radius on standardized features
    delta = 0.5  # threshold for StrongClassificationRobustness

    train_loader, test_loader, model, (mean, std), mode = (
        create_synthetic_tabular_datasets(batch_size=batch_size)
    )
    model = model.to(device)

    logic = logics.LeakyLogic()  # gradients flow past sat boundary
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
        print(
            f"epoch {epoch} TRAIN "
            f"pred_acc={train_info.pred_metric:.3f} "
            f"constr_sec={train_info.constr_sec:.3f} "
            f"pred_loss={train_info.pred_loss:.3f} "
            f"constr_loss={train_info.constr_loss:.3f}"
        )

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
        print(
            f"epoch {epoch}  TEST "
            f"pred_acc={test_info.pred_metric:.3f} "
            f"constr_acc={test_info.constr_acc:.3f} "
            f"constr_sec_self={test_info.constr_sec_self:.3f} "
            f"constr_sec_common={test_info.constr_sec_common:.3f}"
        )


if __name__ == "__main__":
    main()
