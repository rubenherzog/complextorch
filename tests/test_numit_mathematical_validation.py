import math

import torch

from complextorch import build_var_system
from complextorch.measures.backbone import observation_autocovariances
from complextorch.measures.pid import gaussian_pid_from_var
from complextorch.numit import var_total_mutual_information


def _gaussian_mi(covariance: torch.Tensor, n_left: int) -> torch.Tensor:
    left = covariance[:n_left, :n_left]
    right = covariance[n_left:, n_left:]
    return 0.5 * (
        torch.linalg.slogdet(left).logabsdet
        + torch.linalg.slogdet(right).logabsdet
        - torch.linalg.slogdet(covariance).logabsdet
    ) / math.log(2.0)


def test_numit_tmi_matches_predictive_information_identity():
    coefficients = torch.tensor(
        [
            [[0.27, 0.06], [0.03, 0.23]],
            [[0.08, 0.01], [0.02, 0.05]],
        ],
        dtype=torch.float64,
    )
    innovation = torch.tensor([[1.0, 0.17], [0.17, 0.82]], dtype=torch.float64)
    model = build_var_system(coefficients, innovation)
    paper = 0.5 * (
        torch.linalg.slogdet(model.present_covariance).logabsdet
        - torch.linalg.slogdet(innovation).logabsdet
    ) / math.log(2.0)
    pid_total = gaussian_pid_from_var(model, (0,), (1,))["total"]
    tmi = var_total_mutual_information(model)
    torch.testing.assert_close(tmi, paper)
    torch.testing.assert_close(tmi, pid_total)


def test_numit_tmi_matches_explicit_full_history_covariance():
    coefficients = torch.tensor(
        [
            [[0.24, 0.03], [0.04, 0.20]],
            [[0.07, 0.01], [0.00, 0.05]],
            [[0.02, 0.00], [0.01, 0.01]],
        ],
        dtype=torch.float64,
    )
    model = build_var_system(
        coefficients,
        torch.tensor([[1.0, 0.1], [0.1, 0.9]], dtype=torch.float64),
    )
    gamma = observation_autocovariances(model, model.order)[0]
    n, p = model.n_variables, model.order
    joint = torch.empty(((p + 1) * n, (p + 1) * n), dtype=torch.float64)
    for left in range(p + 1):
        for right in range(p + 1):
            delta = right - left
            block = gamma[delta].T if delta >= 0 else gamma[-delta]
            joint[left * n : (left + 1) * n, right * n : (right + 1) * n] = block
    direct = _gaussian_mi(joint, n)
    torch.testing.assert_close(var_total_mutual_information(model).reshape(()), direct)
