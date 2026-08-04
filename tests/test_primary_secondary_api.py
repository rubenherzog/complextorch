import torch

from complextorch import (
    ModelMeasureConfig,
    build_measure_context,
    build_var_system,
    compute_all_model_measures,
    demo_var,
)
from complextorch.measures import (
    PRIMARY_MEASURES,
    SECONDARY_MEASURES,
    primary,
    secondary,
    spectral_mvgc,
    temporal_mvgc,
)


def test_model_first_evaluator_uses_one_shared_autocovariance_sequence():
    coefficients, innovation_covariance = demo_var(n_variables=3, order=2)
    system = build_var_system(coefficients, innovation_covariance)
    config = ModelMeasureConfig(
        frequencies=torch.linspace(0.0, 0.5, 65, dtype=torch.float64),
        autocovariance_max_lag=3,
        cmem_max_lag=5,
        source=(1,),
        target=(0,),
        conditional=(2,),
        phiid_variables=(0, 1),
        phiid_lag=4,
    )
    context = build_measure_context(system, config)
    result = compute_all_model_measures(system, config, context=context)

    assert context.max_lag == 5
    assert result["autocovariances"].data_ptr() == context.autocovariances.data_ptr()
    torch.testing.assert_close(result["gaussian"]["covariance"], system.present_covariance)
    assert {"gaussian", "dynamics", "criticality", "cmem", "frequency", "mvgc", "phiid"}.issubset(result)
    assert "discrete_entropy" not in result
    assert "lempel_ziv_complexity" not in result
    torch.testing.assert_close(
        result["phiid"]["reconstruction"],
        result["phiid"]["total"],
        rtol=1e-8,
        atol=1e-10,
    )


def test_primary_and_secondary_namespaces_are_explicit():
    assert primary.compute_all_model_measures is compute_all_model_measures
    assert callable(primary.temporal_mvgc)
    assert callable(primary.spectral_mvgc)
    assert callable(temporal_mvgc)
    assert callable(spectral_mvgc)
    assert callable(secondary.lempel_ziv_complexity)
    assert callable(secondary.estimate_temporal_mvgc_from_observations)
    assert callable(secondary.estimate_spectral_mvgc_from_observations)
    assert all(metadata["tier"].startswith("primary") for metadata in PRIMARY_MEASURES.values())
    assert all(metadata["tier"] == "secondary" for metadata in SECONDARY_MEASURES.values())
    assert "temporal_mvgc" in PRIMARY_MEASURES
    assert "spectral_mvgc" in PRIMARY_MEASURES
    assert "estimate_temporal_mvgc_from_observations" in SECONDARY_MEASURES
    assert "estimate_spectral_mvgc_from_observations" in SECONDARY_MEASURES
    assert "lempel_ziv_complexity" in SECONDARY_MEASURES
