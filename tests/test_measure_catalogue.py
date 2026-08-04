import math
import torch
from complextorch.measures import *
from complextorch.simulate import demo_var
from complextorch.representations import build_var_system

def system():
    a,q=demo_var(n_variables=3,order=2,dtype=torch.float64)
    return build_var_system(a,q)

def test_gaussian_identities():
    cov=torch.tensor([[2.,.3,.1],[.3,1.5,.2],[.1,.2,1.]],dtype=torch.float64)
    assert torch.allclose(o_information(cov),total_correlation(cov)-dual_total_correlation(cov))
    assert torch.allclose(s_information(cov),total_correlation(cov)+dual_total_correlation(cov))
    joint=torch.block_diag(cov[:2,:2],cov[2:,2:])
    assert abs(float(gaussian_mutual_information(joint,2)))<1e-10
    assert gaussian_entropy(cov).ndim==0

def test_conditional_covariance_and_cmi():
    cov=torch.tensor([[2.,.3,.2],[.3,1.5,.4],[.2,.4,1.]],dtype=torch.float64)
    assert conditional_covariance(cov,2).shape==(2,2)
    assert gaussian_conditional_mutual_information(cov,1,1)>=0

def test_local_mi_mean_matches_global_mi():
    cov=torch.tensor([[1.,.5],[.5,1.]],dtype=torch.float64)
    x=torch.distributions.MultivariateNormal(torch.zeros(2,dtype=torch.float64),cov).sample((50000,))
    local=local_gaussian_mutual_information(x,cov,1)
    assert abs(float(local.mean()-gaussian_mutual_information(cov,1)))<0.03

def test_time_frequency_measures():
    s=system(); f=torch.linspace(0,.5,32,dtype=torch.float64)
    ac=autocovariances(s,4); assert ac.shape==(1,5,3,3)
    assert torch.allclose(ac[:,0],s.present_covariance)
    assert entropy_rate(s).shape==(1,)
    assert predictive_information(s).shape==(1,)
    assert active_information_storage(s).shape==(1,3)
    inv=inverse_transfer_function(s,f); h=transfer_function(s,f)
    eye=torch.eye(3,dtype=h.dtype)
    torch.testing.assert_close(inv@h,eye.expand_as(h),rtol=1e-8,atol=1e-8)
    csd=cross_spectral_density(s,f); torch.testing.assert_close(csd,csd.conj().transpose(-1,-2))
    se=spectral_entropy(s,f); assert bool(((se>=0)&(se<=1+1e-10)).all())

def test_emergence_and_registry():
    s=system(); m=torch.tensor([[1.,1.,0.],[0.,0.,1.]],dtype=torch.float64)/torch.tensor([[math.sqrt(2.)],[1.]])
    out=emergence_measures(s,m)
    assert {'psi','delta','gamma'}<=set(out)
    required={'gaussian_entropy','mutual_information','conditional_mutual_information','conditional_covariance','total_correlation','dual_total_correlation','o_information','s_information','local_gaussian_mi','autocovariances','entropy_rate','predictive_information','active_information_storage','transfer_function','inverse_transfer_function','cross_spectral_density','spectral_entropy','spectral_radius','stability_margin','dominant_timescale','covariance_amplification','psi','delta','gamma','cmem1_total','cmem3_total','cmem1_curve','cmem3_curve','cmem3_lag'}
    assert required<=set(MEASURE_REGISTRY)
