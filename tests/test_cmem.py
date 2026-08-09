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


def numpy_cmem1_reference(coef, q):
    """Independent CMem1 reference using each component's own p-lag history."""
    p, n, _ = coef.shape
    ac = companion_matrix(torch.from_numpy(coef).unsqueeze(0))[0].numpy()
    process = np.zeros_like(ac)
    process[:n, :n] = q
    sz = solve_discrete_lyapunov(ac, process)
    projection = np.zeros((n, n * p))
    projection[:, :n] = np.eye(n)
    sigma = projection @ sz @ projection.T

    powers = [np.eye(n * p)]
    for _ in range(p):
        powers.append(powers[-1] @ ac)
    gamma = [projection @ power @ sz @ projection.T for power in powers]

    _, ld_sigma = np.linalg.slogdet(sigma)
    _, ld_q = np.linalg.slogdet(q)
    full_ais = 0.5 * (ld_sigma - ld_q) / np.log(2)

    part_ais = []
    for node in range(n):
        history = np.empty((p, p))
        for left in range(p):
            for right in range(p):
                history[left, right] = gamma[abs(right - left)][node, node]
        cross = np.array([gamma[lag][node, node] for lag in range(1, p + 1)])
        joint = np.block([
            [np.array([[sigma[node, node]]]), cross[None, :]],
            [cross[:, None], history],
        ])
        _, ld_joint = np.linalg.slogdet(joint)
        _, ld_hist = np.linalg.slogdet(history)
        part_ais.append(
            0.5 * (np.log(sigma[node, node]) + ld_hist - ld_joint) / np.log(2)
        )
    return full_ais - np.sum(part_ais), np.asarray(part_ais)


def test_cmem1_uses_component_self_history_not_full_multivariate_past():
    coef, q = random_stable_var(1, 3, 2, spectral_radius_target=.75, seed=31)
    system = build_var_system(coef, q)
    result = compute_cmem(system, tau_max=3)
    reference, _ = numpy_cmem1_reference(coef[0].numpy(), q[0].numpy())
    np.testing.assert_allclose(result.cmem1_total.numpy(), [reference], rtol=2e-9, atol=2e-9)
    # Coupled dynamics make CMem1 and CMem3 genuinely different in general.
    assert not torch.allclose(result.cmem1_total, result.cmem3_total, rtol=1e-6, atol=1e-6)


def test_active_information_storage_matches_independent_self_history_reference():
    from complextorch.measures import active_information_storage

    coef, q = random_stable_var(1, 4, 2, spectral_radius_target=.7, seed=32)
    system = build_var_system(coef, q)
    _, reference = numpy_cmem1_reference(coef[0].numpy(), q[0].numpy())
    actual = active_information_storage(system)[0].numpy()
    np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=2e-9)


def test_cmem1_and_cmem3_coincide_for_dynamically_independent_components():
    coefficients = torch.zeros((1, 2, 3, 3), dtype=torch.float64)
    coefficients[0, 0] = torch.diag(torch.tensor([.45, .35, .25], dtype=torch.float64))
    coefficients[0, 1] = torch.diag(torch.tensor([.12, .08, .05], dtype=torch.float64))
    q = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    result = compute_cmem(build_var_system(coefficients, q), tau_max=3)
    torch.testing.assert_close(result.cmem1_total, torch.zeros_like(result.cmem1_total), atol=2e-10, rtol=0)
    torch.testing.assert_close(result.cmem3_total, torch.zeros_like(result.cmem3_total), atol=2e-10, rtol=0)
