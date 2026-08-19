"""Proof of concept for representation-general parametric bootstrap inference.

This deliberately lives at test level before changing the public inference API.
It asks whether the existing fixed-fit -> simulate -> fixed-refit logic can be
applied to VAR, general state-space (N4SID), and innovations-form (Larimore)
estimators while comparing one representation-invariant model measure.
"""

from __future__ import annotations

import copy

import torch

from complextorch import (
    LarimoreStateSpace,
    N4SID,
    VAR,
    as_innovations_state_space,
    build_var_system,
    simulate_var,
)
from complextorch.measures.backbone import predictive_information_from_model


def _fit_system(estimator, observations):
    fitted = copy.deepcopy(estimator).fit(observations)
    if isinstance(fitted, VAR):
        return build_var_system(fitted.coef_, fitted.noise_covariance_)
    return fitted.system_


def _expand_matrix(value: torch.Tensor, batch: int) -> torch.Tensor:
    matrix = torch.as_tensor(value)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.shape[0] == 1:
        matrix = matrix.expand(batch, -1, -1)
    return matrix


def _simulate_innovations(
    system,
    *,
    n_resamples: int,
    n_times: int,
    burnin: int,
    seed: int,
) -> torch.Tensor:
    """Simulate observations from one canonical innovations-form process."""
    innovations = as_innovations_state_space(system)
    a = _expand_matrix(innovations.transition, n_resamples)
    c = _expand_matrix(innovations.observation, n_resamples)
    k = _expand_matrix(innovations.gain, n_resamples)
    v = _expand_matrix(innovations.innovation_covariance, n_resamples)

    radius = torch.linalg.eigvals(a).abs().amax(-1)
    assert bool((radius < 1).all())

    generator = torch.Generator(device=a.device).manual_seed(seed)
    chol = torch.linalg.cholesky(v)
    total = n_times + burnin
    noise = torch.randn(
        (n_resamples, total, c.shape[-2]),
        dtype=a.dtype,
        device=a.device,
        generator=generator,
    )
    noise = torch.einsum("bij,btj->bti", chol, noise)
    state = torch.zeros(
        (n_resamples, a.shape[-1]), dtype=a.dtype, device=a.device
    )
    observations = torch.empty_like(noise)
    for time in range(total):
        innovation = noise[:, time]
        observations[:, time] = torch.einsum("bij,bj->bi", c, state) + innovation
        state = (
            torch.einsum("bij,bj->bi", a, state)
            + torch.einsum("bij,bj->bi", k, innovation)
        )
    return observations[:, burnin:]


def _bootstrap_pi(estimator, observations, *, n_resamples: int, seed: int):
    fitted_system = _fit_system(estimator, observations)
    point = predictive_information_from_model(fitted_system).reshape(())
    samples = _simulate_innovations(
        fitted_system,
        n_resamples=n_resamples,
        n_times=observations.shape[-2],
        burnin=100,
        seed=seed,
    )

    refit = copy.deepcopy(estimator)
    refit.mode = "independent"
    ensemble = _fit_system(refit, samples)
    values = predictive_information_from_model(ensemble).reshape(n_resamples)
    alpha = 0.05
    return torch.stack(
        (
            point,
            torch.quantile(values, alpha),
            torch.quantile(values, 1.0 - alpha),
        )
    )


def _experiment(n_times: int):
    dtype = torch.float64
    coefficients = torch.tensor(
        [[[0.45, 0.10], [-0.05, 0.30]]], dtype=dtype
    )
    covariance = torch.tensor(
        [[0.9, 0.18], [0.18, 0.7]], dtype=dtype
    )
    data = simulate_var(
        coefficients,
        covariance,
        n_times,
        burnin=250,
        seed=2026 + n_times,
    )[0]
    true_system = build_var_system(coefficients, covariance)
    truth = predictive_information_from_model(true_system).reshape(())

    estimators = {
        "var": VAR(order=1, mode="pooled", dtype="float64"),
        "ssm": N4SID(n_states=2, block_rows=6, mode="pooled", dtype="float64"),
        "innovations": LarimoreStateSpace(
            n_states=2, past_horizon=6, future_horizon=6, mode="pooled", dtype="float64"
        ),
    }
    intervals = {
        name: _bootstrap_pi(estimator, data, n_resamples=16, seed=900 + index)
        for index, (name, estimator) in enumerate(estimators.items())
    }
    return truth, intervals


def test_parametric_bootstrap_poc_converges_across_var_ssm_and_innovations():
    """Probe finite- and longer-sample behavior before production changes."""
    short_truth, short = _experiment(1600)
    long_truth, long = _experiment(6400)
    for group in (short, long):
        for name, interval in group.items():
            assert bool(torch.isfinite(interval).all()), (name, interval)
    # Temporary diagnostic assertion: CI logs expose the deterministic numerical
    # experiment so the architecture can be chosen from evidence, not assumption.
    raise AssertionError(
        {
            "short_truth": float(short_truth),
            "short": {name: value.tolist() for name, value in short.items()},
            "long_truth": float(long_truth),
            "long": {name: value.tolist() for name, value in long.items()},
        }
    )
