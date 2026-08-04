import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from complextorch.measures import compute_cmem,total_correlation
from complextorch.representations import build_var_system,companion_matrix
from complextorch.simulate import demo_var,random_stable_var

def numpy_tc(cov):
    sign,logdet=np.linalg.slogdet(cov); assert sign>0; return .5*(np.log2(np.diag(cov)).sum()-logdet/np.log(2))

def numpy_reference(coef,q,tau_max):
    p,n,_=coef.shape; ac=companion_matrix(torch.from_numpy(coef).unsqueeze(0))[0].numpy(); process=np.zeros_like(ac); process[:n,:n]=q; sz=solve_discrete_lyapunov(ac,process); projection=np.zeros((n,n*p)); projection[:,:n]=np.eye(n); sigma=projection@sz@projection.T; total=numpy_tc(q)-numpy_tc(sigma); curve=[]; power=np.eye(n*p); inv=np.linalg.inv(sigma)
    for _ in range(tau_max): power=power@ac; gamma=projection@power@sz@projection.T; curve.append(numpy_tc(sigma-gamma@inv@gamma.T)-numpy_tc(sigma))
    return total,np.asarray(curve)

def test_total_correlation_identity():
    eye=torch.eye(4,dtype=torch.float64).expand(3,-1,-1); torch.testing.assert_close(total_correlation(eye),torch.zeros(3,dtype=torch.float64))

def test_cmem_matches_independent_numpy_reference():
    coef,q=demo_var(order=2); result=compute_cmem(build_var_system(coef,q),tau_max=6); total,curve=numpy_reference(coef[0].numpy(),q[0].numpy(),6)
    np.testing.assert_allclose(result.cmem3_total.numpy(),[total],rtol=1e-11,atol=1e-11); np.testing.assert_allclose(result.cmem3_curve[0].numpy(),curve,rtol=1e-10,atol=1e-10)

def test_chain_rule_and_var1_curve_identity():
    coef,q=random_stable_var(5,4,3,spectral_radius_target=.8,seed=20); result=compute_cmem(build_var_system(coef,q),tau_max=4); torch.testing.assert_close(result.cmem3_lag.sum(-1),result.cmem3_total,rtol=2e-8,atol=2e-8)
    coef1,q1=random_stable_var(4,3,1,spectral_radius_target=.8,seed=21); result1=compute_cmem(build_var_system(coef1,q1),tau_max=2); torch.testing.assert_close(result1.cmem3_curve[:,0],result1.cmem3_total,rtol=1e-9,atol=1e-9)
