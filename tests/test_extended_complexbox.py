import torch
from complextorch.control import solve_dare, innovations_form, reduce_state_space, dynamical_dependence, stochastic_interaction
from complextorch.state_space import kalman_filter, kalman_smoother, N4SID, LinearGaussianEM
from complextorch.representations import StateSpaceModel, build_var_system
from complextorch.simulate import simulate_var
from complextorch.measures import temporal_mvgc, pairwise_spectral_gc, discrete_entropy, discrete_mutual_information, discrete_total_correlation, lempel_ziv_complexity, gaussian_phiid_mmi


def latent_system():
    dtype=torch.float64
    return StateSpaceModel(torch.tensor([[.8,.1],[0.,.65]],dtype=dtype),torch.tensor([[1.,0.],[.2,1.]],dtype=dtype),.05*torch.eye(2,dtype=dtype),.1*torch.eye(2,dtype=dtype),state_covariance=torch.eye(2,dtype=dtype))


def simulate_ssm(system,n=400):
    g=torch.Generator().manual_seed(0); d=2; x=torch.zeros(n,d,dtype=torch.float64); y=torch.zeros(n,d,dtype=torch.float64)
    lq=torch.linalg.cholesky(system.process_covariance); lr=torch.linalg.cholesky(system.observation_covariance)
    for t in range(1,n):
        x[t]=system.transition@x[t-1]+lq@torch.randn(d,generator=g,dtype=x.dtype)
        y[t]=system.observation@x[t]+lr@torch.randn(d,generator=g,dtype=x.dtype)
    return y


def test_control_kalman_n4sid_em():
    system=latent_system(); y=simulate_ssm(system)
    assert torch.linalg.eigvalsh(solve_dare(system.transition,system.observation,system.process_covariance,system.observation_covariance)).min()>0
    assert innovations_form(system).covariance.shape==(2,2)
    assert reduce_state_space(system,[0]).observation.shape==(1,2)
    assert torch.isfinite(dynamical_dependence(system)) and torch.isfinite(stochastic_interaction(system,[[0],[1]]))
    assert kalman_filter(y,system).filtered_mean.shape==(400,2)
    assert kalman_smoother(y,system).smoothed_mean.shape==(400,2)
    assert N4SID(2,8).fit(y).system_.transition.shape==(2,2)
    assert len(LinearGaussianEM(system,2).fit(y).log_likelihood_history_)==2


def test_mvgc_discrete_lz_phid():
    a=torch.zeros((1,2,2,2),dtype=torch.float64); a[0,0]=torch.tensor([[.5,0.],[.45,.4]],dtype=torch.float64); a[0,1]=torch.tensor([[.1,0.],[0.,.05]],dtype=torch.float64); q=torch.eye(2,dtype=torch.float64).unsqueeze(0)
    x=simulate_var(a,q,3000,burnin=500,seed=4)[0]
    assert temporal_mvgc(x,2,0,1)>temporal_mvgc(x,2,1,0)
    assert torch.isfinite(pairwise_spectral_gc(build_var_system(a,q),0,1,torch.linspace(0,.5,32,dtype=torch.float64))).all()
    bits=torch.tensor([0,0,1,1]*50); assert abs(discrete_entropy(bits)-1)<1e-12; assert abs(discrete_mutual_information(bits,bits)-1)<1e-12
    assert abs(discrete_total_correlation(torch.stack([bits,bits],1))-1)<1e-12
    assert lempel_ziv_complexity(torch.zeros(100,dtype=torch.long))<lempel_ziv_complexity(torch.arange(100)%2)
    cov=torch.tensor([[1.,.2,.4,.1],[.2,1.,.1,.35],[.4,.1,1.,.25],[.1,.35,.25,1.]],dtype=torch.float64); out=gaussian_phiid_mmi(cov)
    torch.testing.assert_close(out['redundant']+out['unique_x']+out['unique_y']+out['synergistic'],out['total'])
