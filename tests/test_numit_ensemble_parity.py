import math

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq

from complextorch.measures.pid import gaussian_pid_from_var
from complextorch.numit import _match_tmi_by_spectral_radius


def _reference(coefficients, covariance, target):
    n = covariance.shape[0]

    def scale(radius):
        old = np.max(np.abs(np.linalg.eigvals(coefficients[0])))
        return coefficients * (radius / old)

    def state(radius):
        a = scale(radius)[0]
        return a, solve_discrete_lyapunov(a, covariance)

    def tmi(radius):
        _, gamma = state(radius)
        return 0.5 * (
            np.linalg.slogdet(gamma)[1] - np.linalg.slogdet(covariance)[1]
        ) / math.log(2.0)

    radius = brentq(lambda r: tmi(r) - target, 1e-10, 1.0 - 1e-8)
    a, gamma = state(radius)
    cross = gamma @ a.T
    joint = np.block([[gamma, cross], [cross.T, gamma]])
    target_idx = list(range(n, 2 * n))

    def mi(source):
        idx = source + target_idx
        block = joint[np.ix_(idx, idx)]
        k = len(source)
        return 0.5 * (
            np.linalg.slogdet(block[:k, :k])[1]
            + np.linalg.slogdet(block[k:, k:])[1]
            - np.linalg.slogdet(block)[1]
        ) / math.log(2.0)

    i0, i1 = mi([0]), mi([1])
    total = mi([0, 1])
    red = min(i0, i1)
    return radius, np.array(
        [red, i0 - red, i1 - red, total - i0 - i1 + red]
    )


def test_numit_ensemble_parity():
    rng = np.random.default_rng(903)
    coefficients = rng.standard_normal((2, 1, 2, 2))
    normal = rng.standard_normal((2, 2, 3))
    covariance = normal @ np.swapaxes(normal, -1, -2)
    target = torch.full((2,), 2.0, dtype=torch.float64)

    model, radius = _match_tmi_by_spectral_radius(
        torch.tensor(coefficients, dtype=torch.float64),
        torch.tensor(covariance, dtype=torch.float64),
        target,
        base=2.0,
    )
    pid = gaussian_pid_from_var(model, (0,), (1,), redundancy="mmi")
    atoms = torch.stack(
        [
            pid["redundant"],
            pid["unique_source0"],
            pid["unique_source1"],
            pid["synergistic"],
        ],
        -1,
    ).numpy()
    reference = [
        _reference(coefficients[index], covariance[index], 2.0)
        for index in range(2)
    ]

    np.testing.assert_allclose(
        radius.numpy(), [value[0] for value in reference], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        atoms,
        np.stack([value[1] for value in reference]),
        rtol=1e-9,
        atol=1e-10,
    )
