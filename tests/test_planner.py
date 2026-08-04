import torch
from complextorch.measures import DynamicalMeasures
from complextorch.representations import build_var_system
from complextorch.simulate import random_stable_var

def test_measure_planner_shapes_and_shared_cmem():
    coef,q=random_stable_var(6,4,2,spectral_radius_target=.82,seed=77); system=build_var_system(coef,q)
    values=DynamicalMeasures(['spectral_radius','dominant_timescale','cmem3_total','cmem3_curve'],tau_max=5)(system)
    assert values['spectral_radius'].shape==(6,) and values['dominant_timescale'].shape==(6,) and values['cmem3_total'].shape==(6,) and values['cmem3_curve'].shape==(6,5)
    assert torch.isfinite(values['cmem3_curve']).all()
