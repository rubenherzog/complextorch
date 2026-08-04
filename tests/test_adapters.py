import torch
from complextorch.adapters import from_complexbox_timeseries,from_complexbox_var,to_complexbox_timeseries,to_complexbox_var

def test_complexbox_timeseries_roundtrip():
    x=torch.arange(3*20*4).reshape(3,20,4); canonical=from_complexbox_timeseries(x)
    assert canonical.shape==(4,20,3); assert torch.equal(to_complexbox_timeseries(canonical,squeeze_single=False),x)

def test_complexbox_var_roundtrip():
    a=torch.arange(3*3*2).reshape(3,3,2); canonical=from_complexbox_var(a)
    assert canonical.shape==(1,2,3,3); assert torch.equal(to_complexbox_var(canonical),a)
