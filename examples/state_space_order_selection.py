"""Minimal Bauer/Larimore state-space order-selection example."""

import torch

from complextorch import LarimoreStateSpaceOrder


def main() -> None:
    """Estimate a pooled full-model state dimension from batched trajectories."""

    generator = torch.Generator().manual_seed(20260806)
    observations = torch.randn(
        5,
        500,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    estimator = LarimoreStateSpaceOrder(
        past_horizon=10,
        mode="pooled",
        dtype="float64",
    ).fit(observations)

    print("Selected state dimension:", int(estimator.best_order_))
    print("Candidate dimensions:", estimator.candidate_orders_.tolist())
    print("Bauer SVC:", estimator.criterion_.tolist())


if __name__ == "__main__":
    main()
