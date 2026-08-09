import math

import torch

from complextorch import build_var_system
from complextorch.measures.backbone import observation_autocovariances
from complextorch.measures.pid import gaussian_pid, gaussian_pid_from_var, var_past_future_covariance
from complextorch.numit import (
    _match_tmi_by_spectral_radius,
    _var_decay_to_radius,
    var_total_mutual_information,
)


def _direct_gaussian_mi(covariance: torch.Tensor, n_left: int) -> torch.Tensor:
    """Independent Gaussian MI reference from block log determinants."""
    left = covariance[..., :n_left, :n_left]
    right = covariance[..., n_left:, n_left:]
    return 0.5 * (
        torch.linalg.slogdet(left).logabsdet
        + torch.linalg.slogdet(right).logabsdet
        - torch.linalg.slogdet(covariance).logabsdet
    ) / math.log(2.0)


def _direct_dep_redundancy(
    covariance: torch.Tensor, n_source0: int, n_source1: int
) -> torch.Tensor:
    """Independent transcription of Kay-Ince / NuMIT calc_pi_Idep_mvn."""
    n_target = covariance.shape[-1] - n_source0 - n_source1
    x = slice(0, n_source0)
    y = slice(n_source0, n_source0 + n_source1)
    z = slice(n_source0 + n_source1, covariance.shape[-1])

    cxx, cyy, czz = covariance[x, x], covariance[y, y], covariance[z, z]
    cxy, cxz, cyz = covariance[x, y], covariance[x, z], covariance[y, z]
    lx, ly, lz = torch.linalg.cholesky(cxx), torch.linalg.cholesky(cyy), torch.linalg.cholesky(czz)

    # MATLAB chol is upper triangular and the reference uses pinv(chol(C))'.
    # With lower Cholesky factors this is L^{-1} C L^{-T}.
    p = torch.linalg.solve_triangular(lx, cxy, upper=False)
    p = torch.linalg.solve_triangular(ly, p.T, upper=False).T
    q = torch.linalg.solve_triangular(lx, cxz, upper=False)
    q = torch.linalg.solve_triangular(lz, q.T, upper=False).T
    r = torch.linalg.solve_triangular(ly, cyz, upper=False)
    r = torch.linalg.solve_triangular(lz, r.T, upper=False).T

    ix = _direct_gaussian_mi(
        torch.cat(
            [torch.cat([cxx, cxz], -1), torch.cat([cxz.T, czz], -1)], -2
        ),
        n_source0,
    )
    iy = _direct_gaussian_mi(
        torch.cat(
            [torch.cat([cyy, cyz], -1), torch.cat([cyz.T, czz], -1)], -2
        ),
        n_source1,
    )

    def half_log2det(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.linalg.slogdet(matrix).logabsdet / math.log(2.0)

    ex = torch.eye(n_source0, dtype=covariance.dtype)
    ey = torch.eye(n_source1, dtype=covariance.dtype)
    ez = torch.eye(n_target, dtype=covariance.dtype)
    b = ix
    rq = r @ q.T
    i = (
        half_log2det(ey - rq @ rq.T)
        - half_log2det(ez - q.T @ q)
        - half_log2det(ez - r.T @ r)
        - iy
    )
    standardized = torch.cat(
        [
            torch.cat([ex, p, q], -1),
            torch.cat([p.T, ey, r], -1),
            torch.cat([q.T, r.T, ez], -1),
        ],
        -2,
    )
    k = half_log2det(ey - p.T @ p) - half_log2det(standardized) - iy
    unique0 = torch.stack([b, i, k]).min()
    return ix - unique0


def test_numit_tmi_is_exact_var_past_future_mutual_information():
    """Paper TMI I(X_t; X_{t-1:t-p}) equals ComplexTorch predictive information."""
    for coefficients in (
        torch.tensor([[[0.31, 0.07], [0.02, 0.26]]], dtype=torch.float64),
        torch.tensor(
            [
                [[0.27, 0.06], [0.03, 0.23]],
                [[0.08, 0.01], [0.02, 0.05]],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [[0.21, 0.05], [0.02, 0.19]],
                [[0.06, 0.01], [0.01, 0.04]],
                [[0.02, 0.00], [0.00, 0.02]],
            ],
            dtype=torch.float64,
        ),
    ):
        innovation_covariance = torch.tensor(
            [[1.0, 0.17], [0.17, 0.82]], dtype=torch.float64
        )
        model = build_var_system(coefficients, innovation_covariance)

        joint, n0, n1 = var_past_future_covariance(model, (0,), (1,))
        direct = _direct_gaussian_mi(joint, n0 + n1)
        paper_shortcut = 0.5 * (
            torch.linalg.slogdet(model.present_covariance).logabsdet
            - torch.linalg.slogdet(innovation_covariance).logabsdet
        ) / math.log(2.0)
        complextorch_tmi = var_total_mutual_information(model)
        pid_total = gaussian_pid_from_var(model, (0,), (1,))["total"]

        torch.testing.assert_close(complextorch_tmi, direct, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(complextorch_tmi, paper_shortcut, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(complextorch_tmi, pid_total, rtol=1e-10, atol=1e-12)


def test_numit_tmi_matches_explicit_block_toeplitz_history_covariance():
    """Validate the complete p-lag source covariance independently of PID assembly."""
    coefficients = torch.tensor(
        [
            [[0.24, 0.03], [0.04, 0.20]],
            [[0.07, 0.01], [0.00, 0.05]],
            [[0.02, 0.00], [0.01, 0.01]],
        ],
        dtype=torch.float64,
    )
    model = build_var_system(coefficients, torch.tensor([[1.0, 0.1], [0.1, 0.9]], dtype=torch.float64))
    gamma = observation_autocovariances(model, model.order)[0]
    n, p = model.n_variables, model.order
    joint = torch.empty(((p + 1) * n, (p + 1) * n), dtype=torch.float64)
    for left in range(p + 1):
        for right in range(p + 1):
            delta = right - left
            block = gamma[delta].T if delta >= 0 else gamma[-delta]
            joint[left * n : (left + 1) * n, right * n : (right + 1) * n] = block
    direct = _direct_gaussian_mi(joint, n)
    torch.testing.assert_close(var_total_mutual_information(model), direct, rtol=1e-10, atol=1e-12)


def test_gaussian_dep_matches_independent_numit_reference_equations():
    covariance = torch.tensor(
        [
            [1.3, 0.15, 0.18, 0.42, 0.10],
            [0.15, 1.1, 0.12, 0.20, 0.35],
            [0.18, 0.12, 1.0, 0.28, 0.16],
            [0.42, 0.20, 0.28, 1.2, 0.14],
            [0.10, 0.35, 0.16, 0.14, 1.0],
        ],
        dtype=torch.float64,
    )
    # Ensure the hand-written example is SPD before comparing implementations.
    torch.linalg.cholesky(covariance)
    expected = _direct_dep_redundancy(covariance, 2, 1)
    result = gaussian_pid(covariance, 2, 1, redundancy="dep")
    torch.testing.assert_close(result["redundant"], expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(result["reconstruction"], result["total"], rtol=1e-12, atol=1e-12)


def test_tmi_is_monotone_in_numit_spectral_radius_for_random_var2_shapes():
    generator = torch.Generator().manual_seed(91)
    raw = torch.randn((8, 2, 4, 4), generator=generator, dtype=torch.float64)
    draw = torch.randn((8, 4, 5), generator=generator, dtype=torch.float64)
    covariance = draw @ draw.transpose(-1, -2)
    values = []
    for radius in torch.linspace(0.05, 0.95, 19, dtype=torch.float64):
        model = build_var_system(
            _var_decay_to_radius(raw, radius.expand(raw.shape[0])), covariance
        )
        values.append(var_total_mutual_information(model))
    curve = torch.stack(values, 0)
    assert bool(((curve[1:] - curve[:-1]) > -1e-10).all())


def test_tmi_matching_is_accurate_in_float32_and_float64():
    for dtype, atol in ((torch.float64, 1e-8), (torch.float32, 5e-4)):
        generator = torch.Generator().manual_seed(19)
        raw = torch.randn((12, 2, 3, 3), generator=generator, dtype=dtype)
        draw = torch.randn((12, 3, 4), generator=generator, dtype=dtype)
        covariance = draw @ draw.transpose(-1, -2)
        target = torch.full((12,), 1.25, dtype=dtype)
        matched, _ = _match_tmi_by_spectral_radius(raw, covariance, target, base=2.0)
        torch.testing.assert_close(
            var_total_mutual_information(matched), target, rtol=5e-4 if dtype == torch.float32 else 1e-7, atol=atol
        )
