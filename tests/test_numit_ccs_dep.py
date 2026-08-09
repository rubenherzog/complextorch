import pytest
import torch

from complextorch import build_var_system
from complextorch.numit import numit_pid_var


@pytest.mark.parametrize("redundancy", ["dep", "ccs"])
def test_numit_ccs_dep_small_ensemble(redundancy):
    coefficients = torch.tensor(
        [[[0.34, 0.09], [0.05, 0.28]]], dtype=torch.float64
    )
    covariance = torch.tensor(
        [[1.0, 0.12], [0.12, 0.85]], dtype=torch.float64
    )
    model = build_var_system(coefficients, covariance)
    result = numit_pid_var(
        model,
        (0,),
        (1,),
        redundancy=redundancy,
        n_null=8,
        seed=41,
        ccs_qmc_samples=256,
    )

    torch.testing.assert_close(
        result.null_tmi,
        result.target_tmi.expand_as(result.null_tmi),
        rtol=1e-7,
        atol=1e-8,
    )
    torch.testing.assert_close(
        result.observed["reconstruction"], result.observed["total"]
    )
    for name in ("redundant", "unique_source0", "unique_source1", "synergistic"):
        assert bool(torch.isfinite(result.null_atoms[name]).all())
        assert 0.0 <= float(result.quantiles[name]) <= 1.0
