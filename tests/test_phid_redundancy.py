import math

import pytest
import torch

from complextorch import build_var_system, phiid_redundancy_from_model
from complextorch.measures.phid import ATOM_LABELS, gaussian_phiid_atoms
from complextorch.measures.primary import past_future_covariance


COVARIANCE = torch.tensor(
    [
        [1.00, 0.15, 0.30, 0.10],
        [0.15, 1.00, 0.05, 0.25],
        [0.30, 0.05, 1.00, 0.20],
        [0.10, 0.25, 0.20, 1.00],
    ],
    dtype=torch.float64,
)


def _atoms(result):
    return torch.stack([result[name] for name in ATOM_LABELS], dim=-1)


@pytest.mark.parametrize("redundancy", ["mmi", "ccs", "idep_a", "idep_b"])
def test_all_redundancies_reconstruct_total(redundancy):
    kwargs = {"ccs_qmc_samples": 256} if redundancy == "ccs" else {}
    result = gaussian_phiid_atoms(COVARIANCE, redundancy=redundancy, **kwargs)
    assert _atoms(result).shape[-1] == 16
    assert torch.isfinite(_atoms(result)).all()
    torch.testing.assert_close(
        result["reconstruction"], result["total"], rtol=1e-11, atol=1e-11
    )


def test_mmi_matches_previous_product_lattice_values():
    # Regression values from the former MMI-only product-lattice implementation.
    expected = torch.tensor(
        [
            0.001785353100426254,
            0.0,
            0.005387674487580277,
            0.03928741099435271,
            0.005387674487580277,
            0.06078190035002611,
            -0.005387674487580277,
            -0.03795857452782684,
            0.0,
            0.0,
            0.03928741099435271,
            -0.03928741099435271,
            0.04238289722183206,
            -0.04238289722183206,
            -0.03928741099435271,
            0.08282885429485692,
        ],
        dtype=torch.float64,
    )
    actual = _atoms(gaussian_phiid_atoms(COVARIANCE, redundancy="mmi"))
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)


def test_ccs_matches_pinned_imperial_double_redundancy_reference():
    # Independent deterministic evaluation of the local equations in
    # Imperial-MIND-lab/integrated-info-decomp@6c5f2e9, using 512 Sobol nodes.
    result = gaussian_phiid_atoms(
        COVARIANCE, redundancy="ccs", ccs_qmc_samples=512
    )
    torch.testing.assert_close(
        result["red_to_red"],
        torch.tensor(0.0070837770283652245, dtype=torch.float64),
        rtol=2e-11,
        atol=2e-13,
    )
    # These sums are the forward single-target CCS redundancies R_xyta/R_xytb.
    torch.testing.assert_close(
        result["red_to_red"] + result["red_to_unq0"],
        torch.tensor(0.006236336055257821, dtype=torch.float64),
        rtol=2e-11,
        atol=2e-13,
    )
    torch.testing.assert_close(
        result["red_to_red"] + result["red_to_unq1"],
        torch.tensor(0.010983992296046484, dtype=torch.float64),
        rtol=2e-11,
        atol=2e-13,
    )


def test_ccs_is_deterministic_for_fixed_model_and_quadrature():
    first = gaussian_phiid_atoms(
        COVARIANCE, redundancy="ccs", ccs_qmc_samples=256
    )
    second = gaussian_phiid_atoms(
        COVARIANCE, redundancy="ccs", ccs_qmc_samples=256
    )
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)


def test_gaussian_idep_matches_kay_ince_reference_fixture():
    # Independently evaluated from robince/partial-info-decomp
    # calc_pi_Idep_mvn.m at commit 32207164741b9e3ba86cec225c09b4b617681e93.
    result = gaussian_phiid_atoms(COVARIANCE, redundancy="idep_a")
    red_future0 = result["red_to_red"] + result["red_to_unq0"]
    red_future1 = result["red_to_red"] + result["red_to_unq1"]
    torch.testing.assert_close(
        red_future0,
        torch.tensor(0.001785353100426254, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        red_future1,
        torch.tensor(0.004168415273311892, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_idep_directional_closures_are_time_reverse_duals():
    order = torch.tensor([2, 3, 0, 1])
    reversed_covariance = COVARIANCE.index_select(0, order).index_select(1, order)
    forward = gaussian_phiid_atoms(COVARIANCE, redundancy="idep_a")
    backward_reversed = gaussian_phiid_atoms(
        reversed_covariance, redundancy="idep_b"
    )
    torch.testing.assert_close(
        forward["red_to_red"],
        backward_reversed["red_to_red"],
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("redundancy", ["mmi", "ccs", "idep_a", "idep_b"])
def test_batched_matches_explicit_model_loop(redundancy):
    second = COVARIANCE.clone()
    second[0, 2] = second[2, 0] = 0.20
    batch = torch.stack([COVARIANCE, second])
    kwargs = {"ccs_qmc_samples": 128} if redundancy == "ccs" else {}
    batched = gaussian_phiid_atoms(batch, redundancy=redundancy, **kwargs)
    loop = [
        gaussian_phiid_atoms(item, redundancy=redundancy, **kwargs) for item in batch
    ]
    for key in batched:
        expected = torch.stack([item[key] for item in loop])
        torch.testing.assert_close(batched[key], expected, rtol=2e-10, atol=2e-11)


def test_model_first_entrypoint_uses_exact_model_autocovariance():
    coefficients = torch.tensor(
        [[[0.45, 0.08], [0.03, 0.35]]], dtype=torch.float64
    )
    innovations = torch.tensor([[0.20, 0.02], [0.02, 0.15]], dtype=torch.float64)
    model = build_var_system(coefficients, innovations)
    covariance = past_future_covariance(model, (0, 1), lag=1)
    direct = gaussian_phiid_atoms(covariance, redundancy="idep_a")
    model_first = phiid_redundancy_from_model(
        model, (0, 1), lag=1, redundancy="idep_a"
    )
    for key in direct:
        torch.testing.assert_close(model_first[key], direct[key], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("redundancy", ["mmi", "idep_a", "idep_b"])
def test_logarithm_base_conversion(redundancy):
    bits = gaussian_phiid_atoms(COVARIANCE, redundancy=redundancy, base=2.0)
    nats = gaussian_phiid_atoms(COVARIANCE, redundancy=redundancy, base=math.e)
    for key in bits:
        torch.testing.assert_close(
            nats[key], bits[key] * math.log(2.0), rtol=1e-11, atol=1e-11
        )


@pytest.mark.parametrize("redundancy", ["mmi", "idep_a", "idep_b"])
def test_float32_agrees_with_float64(redundancy):
    reference = gaussian_phiid_atoms(COVARIANCE, redundancy=redundancy)
    reduced = gaussian_phiid_atoms(COVARIANCE.float(), redundancy=redundancy)
    torch.testing.assert_close(
        _atoms(reduced).double(), _atoms(reference), rtol=3e-4, atol=3e-5
    )


def test_independent_past_future_has_zero_information():
    for redundancy in ("mmi", "idep_a", "idep_b"):
        result = gaussian_phiid_atoms(
            torch.eye(4, dtype=torch.float64), redundancy=redundancy
        )
        torch.testing.assert_close(
            _atoms(result), torch.zeros(16, dtype=torch.float64), rtol=0.0, atol=1e-12
        )
        torch.testing.assert_close(
            result["total"], torch.zeros((), dtype=torch.float64), rtol=0.0, atol=1e-12
        )


def test_varley_discrete_redundancy_is_not_silently_gaussianized():
    with pytest.raises(NotImplementedError, match="discrete random variables"):
        gaussian_phiid_atoms(COVARIANCE, redundancy="varley")


def test_invalid_inputs_are_explicit():
    with pytest.raises(ValueError, match="at least 32"):
        gaussian_phiid_atoms(COVARIANCE, redundancy="ccs", ccs_qmc_samples=16)
    with pytest.raises(ValueError, match="positive and different from one"):
        gaussian_phiid_atoms(COVARIANCE, base=1.0)
    with pytest.raises(ValueError, match="redundancy must be one of"):
        gaussian_phiid_atoms(COVARIANCE, redundancy="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_matches_cpu_for_analytic_redundancies():
    for redundancy in ("mmi", "idep_a", "idep_b"):
        cpu = gaussian_phiid_atoms(COVARIANCE, redundancy=redundancy)
        gpu = gaussian_phiid_atoms(COVARIANCE.cuda(), redundancy=redundancy)
        torch.testing.assert_close(
            _atoms(gpu).cpu(), _atoms(cpu), rtol=1e-10, atol=1e-10
        )
