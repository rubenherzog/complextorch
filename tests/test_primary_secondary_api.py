import torch

from complextorch import ModelMeasureConfig, build_var_system, compute_all_model_measures, demo_var
from complextorch.measures import PRIMARY_MEASURES, SECONDARY_MEASURES, primary, secondary


def test_model_first_evaluator_uses_model_covariance_and_excludes_empirical_measures():
    coefficients, innovation_covariance = demo_var(n_variables=3, order=2)
    system = build_var_system(coefficients, innovation_covariance)
    config = ModelMeasureConfig(
        frequencies=torch.linspace(0.0, 0.5, 65, dtype=torch.float64),
        max_lag=5,
        cmem_tau_max=5,
        source=(1,),
        target=(0,),
        conditional=(2,),
        phiid_variables=(0, 1),
    )
    result = compute_all_model_measures(system, config)

    torch.testing.assert_close(result["gaussian"]["covariance"], system.present_covariance)
    assert {"gaussian", "dynamics", "criticality", "cmem", "frequency", "mvgc", "phiid"}.issubset(result)
    assert "discrete_entropy" not in result
    assert "lempel_ziv_complexity" not in result
    assert "temporal_mvgc" not in result
    torch.testing.assert_close(
        result["phiid"]["reconstruction"],
        result["phiid"]["total"],
        rtol=1e-8,
        atol=1e-10,
    )


def test_primary_and_secondary_namespaces_are_explicit():
    assert primary.compute_all_model_measures is compute_all_model_measures
    assert callable(secondary.lempel_ziv_complexity)
    assert callable(secondary.temporal_mvgc)
    assert all(metadata["tier"].startswith("primary") for metadata in PRIMARY_MEASURES.values())
    assert all(metadata["tier"] == "secondary" for metadata in SECONDARY_MEASURES.values())
    assert "state_space_temporal_mvgc" in PRIMARY_MEASURES
    assert "temporal_mvgc" in SECONDARY_MEASURES
    assert "lempel_ziv_complexity" in SECONDARY_MEASURES
