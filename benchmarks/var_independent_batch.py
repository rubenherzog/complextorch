"""Benchmark batched-independent VAR fitting against per-series fitting."""

from __future__ import annotations

from time import perf_counter

import torch

from complextorch import VAR


def main() -> None:
    torch.set_num_threads(1)
    observations = torch.randn((128, 300, 5), dtype=torch.float64)
    for solver in ("auto", "lstsq", "pinv", "cholesky", "lwr"):
        kwargs = dict(
            order=3,
            mode="independent",
            solver=solver,
            stability="ignore",
            device="cpu",
            dtype="float64",
        )
        VAR(**kwargs).fit(observations[:8])
        start = perf_counter()
        VAR(**kwargs).fit(observations)
        batched = perf_counter() - start
        start = perf_counter()
        [VAR(**kwargs).fit(row) for row in observations]
        separate = perf_counter() - start
        print(
            f"{solver:9s} batched={batched:.5f}s separate={separate:.5f}s "
            f"speedup={separate / batched:.2f}x"
        )


if __name__ == "__main__":
    main()
