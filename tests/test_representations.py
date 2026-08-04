import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov as scipy_dlyap
from complextorch.linalg import solve_discrete_lyapunov
from complextorch.representations import build_var_system,companion_matrix
from complextorch.simulate import random_stable_var

def test_companion_layout():
    coef=torch.tensor([[[[.2,.1],[0.,.3]],[[.4,0.],[.2,.1]]]],dtype=torch.float64)
    actual=companion_matrix(coef)[0]; expected=torch.tensor([[.2,.1,.4,0.],[0.,.3,.2,.1],[1.,0.,0.,0.],[0.,1.,0.,0.]],dtype=torch.float64)
    torch.testing.assert_close(actual,expected)

def test_lyapunov_doubling_and_direct_match_scipy():
    coef,q=random_stable_var(3,3,2,spectral_radius_target=.75,seed=8); system=build_var_system(coef,q,lyapunov_method='doubling')
    direct,info=solve_discrete_lyapunov(system.companion,system.companion_noise_covariance,method='direct')
    torch.testing.assert_close(system.state_covariance,direct,rtol=1e-10,atol=1e-10); assert info.residual_max<1e-10
    for b in range(3): np.testing.assert_allclose(system.state_covariance[b].numpy(),scipy_dlyap(system.companion[b].numpy(),system.companion_noise_covariance[b].numpy()),rtol=1e-10,atol=1e-10)
